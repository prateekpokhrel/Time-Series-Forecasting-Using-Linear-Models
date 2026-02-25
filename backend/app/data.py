from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf

DataSource = Literal["auto", "local", "yfinance"]

DEFAULT_LOCAL_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass
class PreparedDataset:
    ticker: str
    source: DataSource
    series: pd.Series
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    mean: float
    std: float
    transform: str


def normalize_indian_ticker(raw_ticker: str) -> str:
    ticker = raw_ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker cannot be empty.")
    if ticker.startswith("^"):
        return ticker
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return ticker
    if "." in ticker:
        return ticker
    return f"{ticker}.NS"


def local_symbol_from_ticker(raw_ticker: str) -> str:
    normalized = normalize_indian_ticker(raw_ticker)
    if normalized.startswith("^"):
        return normalized[1:].upper()
    return normalized.split(".")[0].upper()


def list_available_symbols(local_data_dir: str | None = None) -> list[str]:
    data_dir = Path(local_data_dir) if local_data_dir else DEFAULT_LOCAL_DATA_DIR
    if not data_dir.exists():
        return []
    symbols = sorted({p.stem.upper() for p in data_dir.glob("*.csv")})
    return symbols


def _extract_close_series_from_df(df: pd.DataFrame) -> pd.Series:
    if isinstance(df.columns, pd.MultiIndex):
        close_cols = [col for col in df.columns if str(col[0]).strip().lower() == "close"]
        if not close_cols:
            raise ValueError("No Close column found in multi-index CSV.")
        close = pd.to_numeric(df[close_cols[0]], errors="coerce")

        date_cols = [
            col for col in df.columns if str(col[0]).strip().lower() in {"date", "price"}
        ]
        if date_cols:
            dates = pd.to_datetime(df[date_cols[0]], errors="coerce")
        else:
            dates = pd.to_datetime(df.index, errors="coerce")
    else:
        close_col = next((c for c in df.columns if str(c).strip().lower() == "close"), None)
        if close_col is None:
            raise ValueError("No Close column found in CSV.")
        close = pd.to_numeric(df[close_col], errors="coerce")

        date_col = next(
            (c for c in df.columns if str(c).strip().lower() in {"date", "price"}),
            None,
        )
        if date_col is None:
            date_col = df.columns[0]
        dates = pd.to_datetime(df[date_col], errors="coerce")

    series = pd.Series(close.values, index=dates)
    series = series.dropna().astype(float)
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series = series[~series.index.duplicated(keep="last")]
    return series.sort_index()


def _load_close_from_local_csv(csv_path: Path) -> pd.Series:
    attempts = [
        {"skiprows": [1, 2]},
        {},
        {"header": [0, 1], "skiprows": [2]},
    ]
    last_error: Exception | None = None

    for kwargs in attempts:
        try:
            df = pd.read_csv(csv_path, **kwargs)
            series = _extract_close_series_from_df(df)
            if len(series) >= 2:
                return series
        except Exception as exc:  # noqa: PERF203
            last_error = exc

    raise ValueError(f"Could not parse Close prices from '{csv_path}': {last_error}")


def _fetch_close_from_local(raw_ticker: str, local_data_dir: str | None = None) -> pd.Series:
    symbol = local_symbol_from_ticker(raw_ticker)
    data_dir = Path(local_data_dir) if local_data_dir else DEFAULT_LOCAL_DATA_DIR
    candidates = [
        data_dir / f"{symbol}.csv",
        data_dir / f"{normalize_indian_ticker(raw_ticker)}.csv",
    ]
    for path in candidates:
        if path.exists():
            return _load_close_from_local_csv(path)
    raise FileNotFoundError(f"No local CSV found for ticker '{symbol}' in '{data_dir}'.")


def _fetch_close_from_yfinance(raw_ticker: str, period: str, interval: str) -> tuple[str, pd.Series]:
    ticker = normalize_indian_ticker(raw_ticker)
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna().astype(float)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return ticker, close


def fetch_close_series(
    raw_ticker: str,
    period: str = "5y",
    interval: str = "1d",
    data_source: DataSource = "auto",
    local_data_dir: str | None = None,
) -> tuple[str, pd.Series, DataSource]:
    ticker = normalize_indian_ticker(raw_ticker)

    if data_source in {"auto", "local"}:
        try:
            local_series = _fetch_close_from_local(raw_ticker=ticker, local_data_dir=local_data_dir)
            if len(local_series) < 80:
                raise ValueError(
                    f"Local data for '{ticker}' has only {len(local_series)} rows, need >= 80."
                )
            return ticker, local_series, "local"
        except Exception:
            if data_source == "local":
                raise

    yf_ticker, yf_series = _fetch_close_from_yfinance(
        raw_ticker=ticker,
        period=period,
        interval=interval,
    )
    if len(yf_series) < 80:
        raise ValueError(f"Not enough historical rows for '{yf_ticker}'. Got {len(yf_series)} rows.")
    return yf_ticker, yf_series, "yfinance"


def close_to_log_returns(close_values: np.ndarray) -> np.ndarray:
    close_values = close_values.astype(np.float64)
    if np.any(close_values <= 0):
        raise ValueError("Close prices must be positive for log-return transform.")
    returns = np.diff(np.log(close_values))
    return returns.astype(np.float32)


def build_windows(values: np.ndarray, input_len: int, pred_len: int) -> tuple[np.ndarray, np.ndarray]:
    total_window = input_len + pred_len
    if len(values) <= total_window:
        raise ValueError(f"Need more than {total_window} rows, but received {len(values)} rows.")

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for start in range(0, len(values) - total_window + 1):
        x_list.append(values[start : start + input_len])
        y_list.append(values[start + input_len : start + total_window])

    x_arr = np.array(x_list, dtype=np.float32)[..., None]
    y_arr = np.array(y_list, dtype=np.float32)[..., None]
    return x_arr, y_arr


def prepare_dataset(
    raw_ticker: str,
    period: str,
    input_len: int,
    pred_len: int,
    val_ratio: float = 0.2,
    data_source: DataSource = "auto",
    local_data_dir: str | None = None,
) -> PreparedDataset:
    ticker, close_series, source = fetch_close_series(
        raw_ticker=raw_ticker,
        period=period,
        data_source=data_source,
        local_data_dir=local_data_dir,
    )

    close_values = close_series.values.astype(np.float32)
    log_returns = close_to_log_returns(close_values)
    x_all, y_all = build_windows(values=log_returns, input_len=input_len, pred_len=pred_len)

    split_idx = int(len(x_all) * (1 - val_ratio))
    split_idx = max(1, min(split_idx, len(x_all) - 1))

    x_train = x_all[:split_idx]
    y_train = y_all[:split_idx]
    x_val = x_all[split_idx:]
    y_val = y_all[split_idx:]

    mean = float(np.mean(x_train))
    std = float(np.std(x_train))
    if std < 1e-8:
        std = 1.0

    x_train = (x_train - mean) / std
    y_train = (y_train - mean) / std
    x_val = (x_val - mean) / std
    y_val = (y_val - mean) / std

    return PreparedDataset(
        ticker=ticker,
        source=source,
        series=close_series,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        mean=mean,
        std=std,
        transform="log_return",
    )


def make_future_business_days(last_date: pd.Timestamp, horizon: int) -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []
    current = pd.Timestamp(last_date)
    while len(dates) < horizon:
        current = current + pd.Timedelta(days=1)
        if current.weekday() < 5:
            dates.append(current)
    return dates


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV fetching + Technical indicator feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def fetch_ohlcv_dataframe(
    raw_ticker: str,
    period: str = "10y",
    data_source: DataSource = "auto",
    local_data_dir: str | None = None,
) -> tuple[str, pd.DataFrame, DataSource]:
    """
    Fetch OHLCV daily data as a DataFrame.

    Returns (ticker, df, source) where df has columns:
        Close, Open, High, Low, Volume  (all float, index = DatetimeIndex)

    For local CSVs that only contain Close, the other columns will be NaN
    and `compute_feature_matrix` will fall back to univariate mode (zeros).
    """
    ticker = normalize_indian_ticker(raw_ticker)

    # Try yfinance for full OHLCV when allowed
    if data_source in {"auto", "yfinance"}:
        try:
            df = yf.download(
                tickers=ticker,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [str(col[0]).strip() for col in df.columns]
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df = df[~df.index.duplicated(keep="last")].sort_index()
                for col in ["Close", "Open", "High", "Low", "Volume"]:
                    if col not in df.columns:
                        df[col] = np.nan
                df = df[["Close", "Open", "High", "Low", "Volume"]].dropna(subset=["Close"])
                if len(df) >= 80:
                    return ticker, df.astype(float), "yfinance"
        except Exception:
            if data_source == "yfinance":
                raise

    # Fallback: local CSV (Close only — OHLCV columns will be NaN)
    if data_source in {"auto", "local"}:
        close_series = _fetch_close_from_local(raw_ticker=ticker, local_data_dir=local_data_dir)
        df = pd.DataFrame({"Close": close_series})
        for col in ["Open", "High", "Low", "Volume"]:
            df[col] = np.nan
        return ticker, df.astype(float), "local"

    raise ValueError(f"Could not fetch OHLCV data for '{ticker}'.")


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's RSI, same length as input (first `period` values set to 50)."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.zeros(len(close), dtype=np.float64)
    avg_loss = np.zeros(len(close), dtype=np.float64)
    avg_gain[period] = np.mean(gain[1: period + 1])
    avg_loss[period] = np.mean(loss[1: period + 1])

    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    rs = np.where(avg_loss < 1e-10, 100.0, avg_gain / avg_loss)
    rsi = np.where(avg_loss < 1e-10, 100.0, 100.0 - 100.0 / (1.0 + rs))
    rsi[:period] = 50.0
    return rsi.astype(np.float32)


def _macd_histogram(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    """MACD histogram (signal − MACD line), same length as input."""
    def _ema(x: np.ndarray, n: int) -> np.ndarray:
        k = 2.0 / (n + 1)
        out = np.empty(len(x), dtype=np.float64)
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = x[i] * k + out[i - 1] * (1 - k)
        return out

    c = close.astype(np.float64)
    macd_line = _ema(c, fast) - _ema(c, slow)
    return (_ema(macd_line, signal) - macd_line).astype(np.float32)


def _zscore(arr: np.ndarray) -> np.ndarray:
    std = float(np.std(arr))
    if std < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - np.mean(arr)) / std).astype(np.float32)


N_FEATURES = 5   # number of input feature channels


def compute_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Build a [T-1, N_FEATURES] feature matrix from an OHLCV DataFrame.

    Channels:
        0  close_log_return   — primary prediction target
        1  volume_log_return  — volume momentum (z-scored, clipped ±3)
        2  rsi14_norm         — RSI(14) mapped to ~[-0.5, +0.5]
        3  macd_hist_z        — MACD histogram (z-scored)
        4  hl_range_z         — (High−Low)/Close  (z-scored)

    If OHLCV columns are missing, features 1-4 are zero (univariate fallback).
    """
    close = df["Close"].values.astype(np.float64)
    T = len(close)
    ret = np.log(close[1:] / np.maximum(close[:-1], 1e-9)).astype(np.float32)

    has_ohlcv = not df[["High", "Low", "Volume"]].isna().all().all()

    if has_ohlcv:
        volume = df["Volume"].ffill().fillna(1.0).values.astype(np.float64)
        high   = df["High"].fillna(df["Close"]).values.astype(np.float64)
        low    = df["Low"].fillna(df["Close"]).values.astype(np.float64)

        vol_safe = np.maximum(volume, 1.0)
        vol_ret  = np.clip(np.log(vol_safe[1:] / vol_safe[:-1]), -3.0, 3.0)
        vol_ret  = _zscore(vol_ret.astype(np.float32))

        rsi14_ret = (((_rsi(close, 14))[1:] - 50.0) / 100.0).astype(np.float32)

        macd_z = _zscore(_macd_histogram(close)[1:])

        hl = (high[1:] - low[1:]) / np.maximum(close[1:], 1e-9)
        hl_z = _zscore(hl.astype(np.float32))
    else:
        zeros = np.zeros(T - 1, dtype=np.float32)
        vol_ret, rsi14_ret, macd_z, hl_z = zeros, zeros, zeros, zeros

    feat = np.column_stack([ret, vol_ret, rsi14_ret, macd_z, hl_z])
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)