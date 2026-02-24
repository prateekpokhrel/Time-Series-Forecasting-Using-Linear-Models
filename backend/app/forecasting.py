from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import (
    DataSource,
    close_to_log_returns,
    fetch_close_series,
    make_future_business_days,
    normalize_indian_ticker,
)
from .linear_model import Model as LinearModel
from .model import Model as DLinearModel
from .nlinear_model import Model as NLinearModel
from .schemas import ForecastHorizon


@dataclass(frozen=True)
class HorizonConfig:
    key: ForecastHorizon
    interval: str
    period: str
    forecast_steps: int
    history_points: int
    input_len: int
    dlinear_epochs: int
    granularity: str
    band_scale: float


HORIZON_CONFIGS: dict[ForecastHorizon, HorizonConfig] = {
    "1d": HorizonConfig(
        key="1d",
        interval="1m",
        period="7d",
        forecast_steps=375,
        history_points=260,
        input_len=180,
        dlinear_epochs=6,
        granularity="minute",
        band_scale=0.75,
    ),
    "15d": HorizonConfig(
        key="15d",
        interval="60m",
        period="730d",
        forecast_steps=90,
        history_points=220,
        input_len=120,
        dlinear_epochs=8,
        granularity="hour",
        band_scale=1.0,
    ),
    "30d": HorizonConfig(
        key="30d",
        interval="1d",
        period="10y",
        forecast_steps=30,
        history_points=180,
        input_len=90,
        dlinear_epochs=10,
        granularity="day",
        band_scale=1.35,
    ),
}

MARKET_OPEN = time(hour=9, minute=15)
MARKET_CLOSE = time(hour=15, minute=29)
MARKET_HOUR_SLOTS = (10, 11, 12, 13, 14, 15)


def _extract_close(df: pd.DataFrame) -> pd.Series:
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna().astype(float)
    close.index = pd.to_datetime(close.index, utc=True).tz_convert("Asia/Kolkata").tz_localize(None)
    close = close[~close.index.duplicated(keep="last")]
    return close.sort_index()


def _download_intraday_series(raw_ticker: str, period: str, interval: str) -> tuple[str, pd.Series]:
    ticker = normalize_indian_ticker(raw_ticker)
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
        prepost=False,
    )
    if df.empty:
        raise ValueError(f"No yfinance data for {ticker} at interval={interval} period={period}.")

    close = _extract_close(df)
    if len(close) < 200:
        raise ValueError(f"Not enough intraday rows for {ticker}; got {len(close)} rows.")
    return ticker, close


def fetch_series_for_horizon(
    raw_ticker: str,
    horizon: ForecastHorizon,
    data_source: DataSource,
    local_data_dir: str | None,
) -> tuple[str, pd.Series, DataSource, HorizonConfig]:
    cfg = HORIZON_CONFIGS[horizon]

    if horizon in {"1d", "15d"}:
        ticker, series = _download_intraday_series(raw_ticker=raw_ticker, period=cfg.period, interval=cfg.interval)
        return ticker, series, "yfinance", cfg

    ticker, series, source = fetch_close_series(
        raw_ticker=raw_ticker,
        period=cfg.period,
        interval=cfg.interval,
        data_source=data_source,
        local_data_dir=local_data_dir,
    )
    return ticker, series, source, cfg


def _make_supervised_returns(close_values: np.ndarray, input_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_returns = close_to_log_returns(close_values)
    if len(log_returns) <= input_len + 20:
        raise ValueError(
            f"Insufficient history for forecasting: need > {input_len + 20} return points, got {len(log_returns)}."
        )

    x_list: list[np.ndarray] = []
    y_list: list[float] = []
    for idx in range(0, len(log_returns) - input_len):
        x_list.append(log_returns[idx : idx + input_len])
        y_list.append(float(log_returns[idx + input_len]))

    x = np.array(x_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return x, y, log_returns


class TorchOneStepForecaster:
    def __init__(self, model_cls, input_len: int, epochs: int) -> None:
        self.input_len = input_len
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean = 0.0
        self.std = 1.0
        config = SimpleNamespace(seq_len=input_len, pred_len=1, individual=False, enc_in=1)
        self.model = model_cls(config).to(self.device)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, lr: float = 0.001) -> None:
        self.mean = float(np.mean(x_train))
        self.std = float(np.std(x_train))
        if self.std < 1e-8:
            self.std = 1.0

        x_norm = ((x_train - self.mean) / self.std).astype(np.float32)
        y_norm = ((y_train - self.mean) / self.std).astype(np.float32)

        train_x = torch.tensor(x_norm[..., None], dtype=torch.float32)
        train_y = torch.tensor(y_norm[:, None, None], dtype=torch.float32)
        loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                pred = self.model(batch_x)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                optimizer.step()

    def predict_one(self, window: np.ndarray) -> float:
        x_norm = (window.astype(np.float32) - self.mean) / self.std
        x = torch.tensor(x_norm, dtype=torch.float32, device=self.device).view(1, self.input_len, 1)
        self.model.eval()
        with torch.no_grad():
            pred_norm = self.model(x).cpu().numpy().reshape(-1)[0]
        return float(pred_norm * self.std + self.mean)


def _split_train_val(
    x: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = int(len(x) * (1 - val_ratio))
    split = max(24, min(split, len(x) - 24))
    return x[:split], y[:split], x[split:], y[split:]


def _stacking_weights(preds: np.ndarray, target: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(preds, target, rcond=None)
    coef = np.clip(coef, 0.0, None)
    if float(np.sum(coef)) < 1e-12:
        coef = np.ones(preds.shape[1], dtype=np.float32)
    coef = coef / np.sum(coef)
    return coef.astype(np.float32)


def _shape_signal(steps: int, horizon: ForecastHorizon) -> np.ndarray:
    t = np.arange(steps, dtype=np.float32)
    if horizon == "1d":
        base = np.sin(2 * np.pi * t / 41.0) + 0.52 * np.cos(2 * np.pi * t / 87.0)
        edge = 0.72 + 0.58 * (np.abs((t / max(steps - 1, 1)) - 0.5) * 2.0)
        base = base * edge
    elif horizon == "15d":
        base = np.sin(2 * np.pi * t / 10.0) + 0.35 * np.cos(2 * np.pi * t / 23.0)
    else:
        base = np.sin(2 * np.pi * t / 5.0) + 0.28 * np.cos(2 * np.pi * t / 11.0)

    base = base - np.mean(base)
    std = float(np.std(base))
    if std < 1e-8:
        return np.zeros_like(base)
    return (base / std).astype(np.float32)


def _volatility_aware_returns(
    forecast_returns: np.ndarray,
    recent_returns: np.ndarray,
    horizon: ForecastHorizon,
) -> np.ndarray:
    if len(forecast_returns) < 3 or len(recent_returns) < 30:
        return forecast_returns

    recent_std = float(np.std(recent_returns))
    current_std = float(np.std(forecast_returns))
    if recent_std < 1e-9:
        return forecast_returns

    target_factor = {"1d": 0.9, "15d": 0.78, "30d": 0.62}[horizon]
    target_std = recent_std * target_factor
    if current_std >= target_std * 0.85:
        return forecast_returns

    shape = _shape_signal(len(forecast_returns), horizon)
    missing_std = float(np.sqrt(max(target_std * target_std - current_std * current_std, 0.0)))
    adjusted = forecast_returns + shape * missing_std

    clip_base = float(np.quantile(np.abs(recent_returns), 0.985))
    if clip_base > 0:
        adjusted = np.clip(adjusted, -2.2 * clip_base, 2.2 * clip_base)

    # Preserve the long-run projected close while adding short-term up/down moves.
    drift_fix = (float(np.sum(forecast_returns)) - float(np.sum(adjusted))) / len(adjusted)
    adjusted = adjusted + drift_fix
    return adjusted.astype(np.float32)


def _validate_models(
    models: dict[str, TorchOneStepForecaster],
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[np.ndarray, list[str], dict[str, float], float]:
    model_names = list(models.keys())
    preds_by_model: dict[str, np.ndarray] = {}

    for name, model in models.items():
        preds_by_model[name] = np.array([model.predict_one(w) for w in x_val], dtype=np.float32)

    pred_mat = np.column_stack([preds_by_model[name] for name in model_names]).astype(np.float32)
    weights = _stacking_weights(pred_mat, y_val)
    ensemble = pred_mat @ weights

    model_mse: dict[str, float] = {}
    for name in model_names:
        model_mse[name] = float(np.mean((preds_by_model[name] - y_val) ** 2))
    model_mse["stacked"] = float(np.mean((ensemble - y_val) ** 2))

    residual_std = float(np.std(y_val - ensemble))
    return weights, model_names, model_mse, residual_std


def _rollout_forecast_returns(
    models: dict[str, TorchOneStepForecaster],
    model_names: list[str],
    weights: np.ndarray,
    initial_window: np.ndarray,
    steps: int,
) -> np.ndarray:
    rolling = initial_window.copy().astype(np.float32)
    out: list[float] = []

    for _ in range(steps):
        preds = [models[name].predict_one(rolling) for name in model_names]
        stacked = float(np.dot(weights, np.array(preds, dtype=np.float32)))
        out.append(stacked)
        rolling = np.concatenate([rolling[1:], np.array([stacked], dtype=np.float32)])

    return np.array(out, dtype=np.float32)


def _returns_to_prices(last_close: float, returns: np.ndarray) -> np.ndarray:
    return last_close * np.exp(np.cumsum(returns))


def _next_trading_minutes(last_dt: pd.Timestamp, steps: int) -> list[pd.Timestamp]:
    timestamps: list[pd.Timestamp] = []
    current = pd.Timestamp(last_dt).floor("min")

    while len(timestamps) < steps:
        current = current + pd.Timedelta(minutes=1)

        if current.weekday() >= 5:
            next_bday = (current + pd.offsets.BDay(1)).normalize()
            current = next_bday + pd.Timedelta(hours=9, minutes=14)
            continue

        if current.time() < MARKET_OPEN:
            current = current.normalize() + pd.Timedelta(hours=9, minutes=15)
        elif current.time() > MARKET_CLOSE:
            next_bday = (current + pd.offsets.BDay(1)).normalize()
            current = next_bday + pd.Timedelta(hours=9, minutes=14)
            continue

        timestamps.append(current)

    return timestamps


def _next_trading_hours(last_dt: pd.Timestamp, steps: int) -> list[pd.Timestamp]:
    timestamps: list[pd.Timestamp] = []
    current = pd.Timestamp(last_dt)

    while len(timestamps) < steps:
        day = current.normalize()

        if day.weekday() >= 5:
            day = (day + pd.offsets.BDay(1)).normalize()

        emitted = False
        for hour in MARKET_HOUR_SLOTS:
            candidate = day + pd.Timedelta(hours=hour)
            if candidate > current:
                timestamps.append(candidate)
                current = candidate
                emitted = True
                break

        if not emitted:
            current = (day + pd.offsets.BDay(1)).normalize()

    return timestamps


def _format_timestamp(ts: pd.Timestamp, granularity: str) -> str:
    if granularity in {"minute", "hour"}:
        return ts.strftime("%Y-%m-%d %H:%M")
    return ts.strftime("%Y-%m-%d")


def run_stacked_forecast(
    raw_ticker: str,
    horizon: ForecastHorizon,
    data_source: DataSource = "auto",
    local_data_dir: str | None = None,
) -> dict:
    ticker, series, used_source, cfg = fetch_series_for_horizon(
        raw_ticker=raw_ticker,
        horizon=horizon,
        data_source=data_source,
        local_data_dir=local_data_dir,
    )

    close_values = series.values.astype(np.float32)
    x_all, y_all, returns = _make_supervised_returns(close_values=close_values, input_len=cfg.input_len)
    x_train, y_train, x_val, y_val = _split_train_val(x_all, y_all)

    models: dict[str, TorchOneStepForecaster] = {
        "dlinear": TorchOneStepForecaster(DLinearModel, input_len=cfg.input_len, epochs=cfg.dlinear_epochs),
        "linear": TorchOneStepForecaster(LinearModel, input_len=cfg.input_len, epochs=max(4, cfg.dlinear_epochs // 2 + 2)),
        "nlinear": TorchOneStepForecaster(NLinearModel, input_len=cfg.input_len, epochs=max(4, cfg.dlinear_epochs // 2 + 2)),
    }

    for model in models.values():
        model.fit(x_train=x_train, y_train=y_train)

    weights, model_names, model_mse, residual_std = _validate_models(
        models=models,
        x_val=x_val,
        y_val=y_val,
    )
    uniform = np.ones_like(weights, dtype=np.float32) / len(weights)
    weights = (0.82 * weights + 0.18 * uniform).astype(np.float32)
    weights = weights / np.sum(weights)

    forecast_returns = _rollout_forecast_returns(
        models=models,
        model_names=model_names,
        weights=weights,
        initial_window=returns[-cfg.input_len :],
        steps=cfg.forecast_steps,
    )
    forecast_returns = _volatility_aware_returns(
        forecast_returns=forecast_returns,
        recent_returns=returns[-max(cfg.input_len * 2, 120) :],
        horizon=horizon,
    )

    last_close = float(close_values[-1])
    forecast_values = _returns_to_prices(last_close=last_close, returns=forecast_returns)

    upper_returns = forecast_returns + cfg.band_scale * residual_std
    lower_returns = forecast_returns - cfg.band_scale * residual_std
    high_values = _returns_to_prices(last_close=last_close, returns=upper_returns)
    low_values = _returns_to_prices(last_close=last_close, returns=lower_returns)

    if cfg.granularity == "minute":
        future_dates = _next_trading_minutes(last_dt=series.index[-1], steps=cfg.forecast_steps)
    elif cfg.granularity == "hour":
        future_dates = _next_trading_hours(last_dt=series.index[-1], steps=cfg.forecast_steps)
    else:
        future_dates = make_future_business_days(last_date=series.index[-1], horizon=cfg.forecast_steps)

    history_series = series.tail(cfg.history_points)
    history = [
        {"date": _format_timestamp(idx, cfg.granularity), "value": float(value)}
        for idx, value in history_series.items()
    ]

    forecast = []
    for dt, pred, hi, lo in zip(future_dates, forecast_values, high_values, low_values, strict=True):
        row = {
            "date": _format_timestamp(dt, cfg.granularity),
            "value": float(pred),
        }
        if horizon == "30d":
            row["high"] = float(max(hi, pred, lo))
            row["low"] = float(min(hi, pred, lo))
        forecast.append(row)

    projected_close = float(forecast_values[-1])
    projected_change_pct = ((projected_close - last_close) / last_close) * 100 if last_close != 0 else 0.0

    stack_weights = {name: float(w) for name, w in zip(model_names, weights, strict=True)}

    return {
        "ticker": ticker,
        "source": used_source,
        "horizon": horizon,
        "granularity": cfg.granularity,
        "last_close": last_close,
        "projected_close": projected_close,
        "projected_change_pct": float(projected_change_pct),
        "history": history,
        "forecast": forecast,
        "stack_weights": stack_weights,
        "model_mse": model_mse,
    }
