from __future__ import annotations

import math
from datetime import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import (
    DataSource,
    N_FEATURES,
    compute_feature_matrix,
    fetch_ohlcv_dataframe,
    make_future_business_days,
)
from .card_model import CARDModel
from .linear_model import Model as LinearModel
from .model import Model as DLinearModel
from .nlinear_model import Model as NLinearModel
from .patchtst_model import PatchTSTModel
from .schemas import ForecastHorizon

# ─────────────────────────────────────────────────────────────────────────────
# Shared base configuration — ALL horizons use the same daily model
# ─────────────────────────────────────────────────────────────────────────────
_BASE_INPUT_LEN = 90        # lookback window in trading days
_BASE_FORECAST_DAYS = 30    # always generate 30 daily steps
_BASE_PERIOD = "10y"        # fetch 10 years of daily data
_BASE_EPOCHS_DLINEAR = 10
_BASE_EPOCHS_LINEAR = 6

# Market session constants (NSE)
MARKET_OPEN = time(hour=9, minute=15)
MARKET_CLOSE = time(hour=15, minute=29)
MARKET_HOUR_SLOTS = (10, 11, 12, 13, 14, 15)  # 6 hourly slots per trading day

# How many history bars to show per horizon (daily bars, consistent source)
_HISTORY_POINTS: dict[ForecastHorizon, int] = {
    "1d": 60,
    "15d": 120,
    "30d": 180,
}

# Confidence-band width scalar per horizon
_BAND_SCALE: dict[ForecastHorizon, float] = {
    "1d": 0.75,
    "15d": 1.0,
    "30d": 1.35,
}


# ─────────────────────────────────────────────────────────────────────────────
# One-step forecaster (trains one model, predicts one step at a time)
# ─────────────────────────────────────────────────────────────────────────────

class TorchOneStepForecaster:
    def __init__(self, model_cls, input_len: int, epochs: int) -> None:
        self.input_len = input_len
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Normalisation stats computed from close-return (channel 0) only
        self.mean = 0.0
        self.std = 1.0
        # All models trained with N_FEATURES channels, channel-independent
        config = SimpleNamespace(
            seq_len=input_len,
            pred_len=1,
            individual=True,    # each channel gets its own weights
            enc_in=N_FEATURES,
        )
        self.model = model_cls(config).to(self.device)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, lr: float = 0.001) -> None:
        """
        x_train: [N, seq_len, N_FEATURES]  — multi-channel windows
        y_train: [N]                        — close log-return target
        """
        # Normalise using close-channel (ch 0) stats
        close_ch = x_train[:, :, 0]
        self.mean = float(np.mean(close_ch))
        self.std  = float(np.std(close_ch))
        if self.std < 1e-8:
            self.std = 1.0

        # Normalise all channels by the same close-channel stats so scale is consistent
        x_norm = ((x_train - self.mean) / self.std).astype(np.float32)
        y_norm = ((y_train - self.mean) / self.std).astype(np.float32)

        train_x = torch.tensor(x_norm, dtype=torch.float32)             # [N, L, C]
        # Target: [N, 1] — the close return; we extract ch0 of model output in loss
        train_y = torch.tensor(y_norm[:, None], dtype=torch.float32)    # [N, 1]
        loader  = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)

        loss_fn   = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                pred = self.model(batch_x)              # [B, pred_len, C] or [B, pred_len, 1]
                # Supervise only on channel 0 (close return), step 0
                pred_close = pred[:, 0, 0:1]            # [B, 1]
                loss = loss_fn(pred_close, batch_y)
                loss.backward()
                optimizer.step()

    def predict_one(self, window: np.ndarray) -> float:
        """
        window: [seq_len, N_FEATURES]  — multi-channel lookback
        returns: float — predicted close log-return
        """
        x_norm = ((window.astype(np.float32) - self.mean) / self.std)
        x = torch.tensor(x_norm, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, L, C]
        self.model.eval()
        with torch.no_grad():
            pred_norm = self.model(x).cpu().numpy().reshape(-1)[0]
        return float(pred_norm * self.std + self.mean)


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_supervised_features(
    feat_matrix: np.ndarray,
    input_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build supervised learning windows from a [T, N_FEATURES] feature matrix.

    Returns:
        x: [N, input_len, N_FEATURES]  — input windows
        y: [N]                          — close log-return at step t+input_len
    """
    T = len(feat_matrix)
    if T <= input_len + 20:
        raise ValueError(
            f"Insufficient history: need > {input_len + 20} feature rows, got {T}."
        )

    x_list: list[np.ndarray] = []
    y_list: list[float] = []
    for idx in range(T - input_len):
        x_list.append(feat_matrix[idx: idx + input_len])          # [input_len, N_FEATURES]
        y_list.append(float(feat_matrix[idx + input_len, 0]))     # close return (ch 0)

    x = np.array(x_list, dtype=np.float32)    # [N, L, C]
    y = np.array(y_list, dtype=np.float32)    # [N]
    return x, y


def _split_train_val(
    x: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = int(len(x) * (1 - val_ratio))
    split = max(24, min(split, len(x) - 24))
    return x[:split], y[:split], x[split:], y[split:]


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stacking_weights(preds: np.ndarray, target: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(preds, target, rcond=None)
    coef = np.clip(coef, 0.0, None)
    if float(np.sum(coef)) < 1e-12:
        coef = np.ones(preds.shape[1], dtype=np.float32)
    coef = coef / np.sum(coef)
    return coef.astype(np.float32)


def _validate_models(
    models: dict[str, TorchOneStepForecaster],
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[np.ndarray, list[str], dict[str, float], float]:
    model_names = list(models.keys())
    preds_by_model: dict[str, np.ndarray] = {}

    for name, model in models.items():
        preds_by_model[name] = np.array([model.predict_one(w) for w in x_val], dtype=np.float32)

    pred_mat = np.column_stack([preds_by_model[n] for n in model_names]).astype(np.float32)
    weights = _stacking_weights(pred_mat, y_val)
    ensemble = pred_mat @ weights

    model_mse: dict[str, float] = {
        name: float(np.mean((preds_by_model[name] - y_val) ** 2))
        for name in model_names
    }
    model_mse["stacked"] = float(np.mean((ensemble - y_val) ** 2))

    residual_std = float(np.std(y_val - ensemble))
    return weights, model_names, model_mse, residual_std


def _rollout_forecast_returns(
    models: dict[str, TorchOneStepForecaster],
    model_names: list[str],
    weights: np.ndarray,
    initial_window: np.ndarray,   # [input_len, N_FEATURES]
    steps: int,
) -> np.ndarray:
    """
    Autoregressively generate `steps` close log-returns.

    For the non-close feature channels (RSI, Volume, MACD, HL) we propagate
    the last known row forward — a reasonable approximation since indicators
    change slowly relative to a 1-step return prediction horizon.
    """
    rolling = initial_window.copy().astype(np.float32)   # [L, C]
    out: list[float] = []

    for _ in range(steps):
        preds   = [models[name].predict_one(rolling) for name in model_names]
        stacked = float(np.dot(weights, np.array(preds, dtype=np.float32)))
        out.append(stacked)

        # Build new row: update close channel, propagate others
        new_row             = rolling[-1].copy()   # inherit last indicator values
        new_row[0]          = stacked              # update close return
        rolling             = np.vstack([rolling[1:], new_row])

    return np.array(out, dtype=np.float32)


def _returns_to_prices(last_close: float, returns: np.ndarray) -> np.ndarray:
    return (last_close * np.exp(np.cumsum(returns))).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Volatility post-processing
# ─────────────────────────────────────────────────────────────────────────────

def _shape_signal(steps: int) -> np.ndarray:
    t = np.arange(steps, dtype=np.float32)
    base = np.sin(2 * np.pi * t / 5.0) + 0.28 * np.cos(2 * np.pi * t / 11.0)
    base = base - np.mean(base)
    std = float(np.std(base))
    if std < 1e-8:
        return np.zeros_like(base)
    return (base / std).astype(np.float32)


def _volatility_aware_returns(
    forecast_returns: np.ndarray,
    recent_returns: np.ndarray,
) -> np.ndarray:
    """Calibrate forecast volatility to match recent historical regime."""
    if len(forecast_returns) < 3 or len(recent_returns) < 30:
        return forecast_returns

    recent_std = float(np.std(recent_returns))
    current_std = float(np.std(forecast_returns))
    if recent_std < 1e-9:
        return forecast_returns

    target_std = recent_std * 0.62   # 30d target factor (daily model)
    if current_std >= target_std * 0.85:
        return forecast_returns

    shape = _shape_signal(len(forecast_returns))
    missing_std = float(np.sqrt(max(target_std ** 2 - current_std ** 2, 0.0)))
    adjusted = forecast_returns + shape * missing_std

    clip_base = float(np.quantile(np.abs(recent_returns), 0.985))
    if clip_base > 0:
        adjusted = np.clip(adjusted, -2.2 * clip_base, 2.2 * clip_base)

    # Preserve the long-run projected close while adding short-term moves
    drift_fix = (float(np.sum(forecast_returns)) - float(np.sum(adjusted))) / len(adjusted)
    adjusted = adjusted + drift_fix
    return adjusted.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Intraday expansion  (daily close → minute / hourly prices)
# ─────────────────────────────────────────────────────────────────────────────

def _expand_daily_to_minutes(
    prev_close: float,
    target_close: float,
    n_minutes: int = 375,
) -> np.ndarray:
    """
    Synthesize n_minutes intraday prices consistent with a single predicted daily close.

    Uses a linear drift toward target_close plus a sinusoidal intraday wiggle.
    The final bar is hard-anchored to target_close so all horizons share the same
    daily close anchor.
    """
    total_log_return = math.log(max(target_close, 1e-9) / max(prev_close, 1e-9))
    t = np.linspace(0.0, 1.0, n_minutes, dtype=np.float32)

    drift = total_log_return * t
    amplitude = abs(total_log_return) * 0.25
    wiggle = amplitude * np.sin(2.0 * np.pi * t)

    prices = prev_close * np.exp(drift + wiggle)
    prices[-1] = target_close   # hard-anchor final bar
    return prices.astype(np.float32)


def _expand_days_to_hours(
    prev_close: float,
    daily_prices: np.ndarray,
    slots_per_day: int = 6,
) -> np.ndarray:
    """
    Expand N daily closes into N × slots_per_day hourly prices.

    Each day's intraday path drifts from the previous day's close to the current
    day's predicted close. The final bar of every day is anchored exactly to the
    predicted daily close, so all horizons share the same daily price anchors.
    """
    all_prices: list[float] = []
    anchor = float(prev_close)

    for day_close in daily_prices:
        day_close = float(day_close)
        total_return = math.log(max(day_close, 1e-9) / max(anchor, 1e-9))
        t = np.linspace(1.0 / slots_per_day, 1.0, slots_per_day, dtype=np.float32)
        hourly_returns = total_return * t
        # Small mid-day sine pattern for realism
        wiggle = abs(total_return) * 0.12 * np.sin(np.pi * t)
        prices = anchor * np.exp(hourly_returns + wiggle)
        prices[-1] = day_close  # hard-anchor
        all_prices.extend(prices.tolist())
        anchor = day_close

    return np.array(all_prices, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Timestamp generators
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point — COHERENT multi-horizon forecast
# ─────────────────────────────────────────────────────────────────────────────

def run_stacked_forecast(
    raw_ticker: str,
    horizon: ForecastHorizon,
    data_source: DataSource = "auto",
    local_data_dir: str | None = None,
) -> dict:
    """
    Generate a coherent multi-horizon forecast.

    All three horizons (1d, 15d, 30d) share the SAME underlying 30-day daily
    forecast so that e.g. Day-1 @ 12:00 in the '1d' view matches Day-1 @ 12:00
    in the '15d' view — they are both derived from daily_prices[0].

    Pipeline:
        10-year daily data
          → train DLinear + Linear + NLinear (daily returns)
          → NNLS ensemble → 30-day daily rollout
          → expand to intraday granularity per horizon:
              "1d"  : day-1 close → 375 synthetic minute bars
              "15d" : days 1-15 closes → 90 synthetic hourly bars (6/day)
              "30d" : all 30 daily closes, with confidence bands
    """
    # ── 1. Fetch full OHLCV daily data ────────────────────────────────────
    ticker, ohlcv_df, source = fetch_ohlcv_dataframe(
        raw_ticker=raw_ticker,
        period=_BASE_PERIOD,
        data_source=data_source,
        local_data_dir=local_data_dir,
    )

    close_values = ohlcv_df["Close"].values.astype(np.float32)
    series = ohlcv_df["Close"]   # pd.Series — used for timestamps

    # ── 1b. Fix random seed for reproducibility across horizon calls ──────
    _seed = abs(hash(ticker)) % (2 ** 31) ^ int(close_values[-1] * 100) % (2 ** 16)
    torch.manual_seed(_seed)
    np.random.seed(_seed % (2 ** 32))

    # ── 2. Build 5-channel feature matrix and supervised windows ──────────
    feat_matrix = compute_feature_matrix(ohlcv_df)                   # [T-1, N_FEATURES]
    x_all, y_all = _make_supervised_features(feat_matrix, _BASE_INPUT_LEN)
    x_train, y_train, x_val, y_val = _split_train_val(x_all, y_all)

    # ── 3. Train all five models on daily data ─────────────────────────────
    models: dict[str, TorchOneStepForecaster] = {
        "dlinear":  TorchOneStepForecaster(DLinearModel,  input_len=_BASE_INPUT_LEN, epochs=_BASE_EPOCHS_DLINEAR),
        "linear":   TorchOneStepForecaster(LinearModel,   input_len=_BASE_INPUT_LEN, epochs=_BASE_EPOCHS_LINEAR),
        "nlinear":  TorchOneStepForecaster(NLinearModel,  input_len=_BASE_INPUT_LEN, epochs=_BASE_EPOCHS_LINEAR),
        "patchtst": TorchOneStepForecaster(PatchTSTModel, input_len=_BASE_INPUT_LEN, epochs=8),
        "card":     TorchOneStepForecaster(CARDModel,     input_len=_BASE_INPUT_LEN, epochs=8),
    }
    for model in models.values():
        model.fit(x_train=x_train, y_train=y_train)

    # ── 4. Stacking weights from validation ──────────────────────────────
    weights, model_names, model_mse, residual_std = _validate_models(models, x_val, y_val)
    uniform = np.ones_like(weights, dtype=np.float32) / len(weights)
    weights = (0.82 * weights + 0.18 * uniform).astype(np.float32)
    weights = weights / np.sum(weights)

    # ── 5. 30-day daily rollout (multi-channel sliding window) ────────────
    initial_window = feat_matrix[-_BASE_INPUT_LEN:]                  # [L, N_FEATURES]
    # Derive returns for volatility calculation (close channel only)
    returns = feat_matrix[:, 0]                                       # [T-1] close log-returns
    forecast_returns = _rollout_forecast_returns(
        models=models,
        model_names=model_names,
        weights=weights,
        initial_window=initial_window,
        steps=_BASE_FORECAST_DAYS,
    )
    forecast_returns = _volatility_aware_returns(
        forecast_returns=forecast_returns,
        recent_returns=returns[-max(_BASE_INPUT_LEN * 2, 120):],
    )

    last_close = float(close_values[-1])
    daily_prices = _returns_to_prices(last_close, forecast_returns)           # shape: (30,)
    future_days = make_future_business_days(series.index[-1], _BASE_FORECAST_DAYS)

    # ── 6. Confidence bands (used only for 30d, but computed for all) ─────
    band_scale = _BAND_SCALE[horizon]
    high_prices = _returns_to_prices(last_close, forecast_returns + band_scale * residual_std)
    low_prices  = _returns_to_prices(last_close, forecast_returns - band_scale * residual_std)

    # ── 7. Expand to the requested horizon granularity ────────────────────
    granularity: str
    forecast: list[dict]

    if horizon == "1d":
        granularity = "minute"
        # Expand day-0 predicted close into 375 intraday minute bars
        minute_prices = _expand_daily_to_minutes(
            prev_close=last_close,
            target_close=float(daily_prices[0]),
            n_minutes=375,
        )
        future_dates = _next_trading_minutes(series.index[-1], 375)
        forecast = [
            {"date": _format_timestamp(dt, "minute"), "value": float(p)}
            for dt, p in zip(future_dates, minute_prices, strict=True)
        ]
        projected_close = float(daily_prices[0])
        history_series = series.tail(_HISTORY_POINTS["1d"])

    elif horizon == "15d":
        granularity = "hour"
        # Expand first 15 daily closes → 90 hourly bars (6 per day)
        hourly_prices = _expand_days_to_hours(
            prev_close=last_close,
            daily_prices=daily_prices[:15],
            slots_per_day=6,
        )
        future_dates = _next_trading_hours(series.index[-1], len(hourly_prices))
        forecast = [
            {"date": _format_timestamp(dt, "hour"), "value": float(p)}
            for dt, p in zip(future_dates, hourly_prices, strict=True)
        ]
        projected_close = float(daily_prices[14])
        history_series = series.tail(_HISTORY_POINTS["15d"])

    else:  # "30d"
        granularity = "day"
        forecast = []
        for dt, pred, hi, lo in zip(future_days, daily_prices, high_prices, low_prices, strict=True):
            forecast.append({
                "date": _format_timestamp(dt, "day"),
                "value": float(pred),
                "high": float(max(hi, pred, lo)),
                "low":  float(min(hi, pred, lo)),
            })
        projected_close = float(daily_prices[-1])
        history_series = series.tail(_HISTORY_POINTS["30d"])

    # ── 8. History (always daily bars for consistency) ────────────────────
    history = [
        {"date": _format_timestamp(idx, "day"), "value": float(val)}
        for idx, val in history_series.items()
    ]

    projected_change_pct = (
        ((projected_close - last_close) / last_close) * 100
        if last_close != 0 else 0.0
    )
    stack_weights = {name: float(w) for name, w in zip(model_names, weights, strict=True)}

    return {
        "ticker": ticker,
        "source": source,
        "horizon": horizon,
        "granularity": granularity,
        "last_close": last_close,
        "projected_close": projected_close,
        "projected_change_pct": float(projected_change_pct),
        "history": history,
        "forecast": forecast,
        "stack_weights": stack_weights,
        "model_mse": model_mse,
    }
