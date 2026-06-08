from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_series(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce").dropna()


def _latest(series: pd.Series) -> float:
    series = _clean_series(series)
    if series.empty:
        return np.nan
    return float(series.iloc[-1])


def _column(df: pd.DataFrame, name: str) -> pd.Series | None:
    lower_map = {str(c).lower(): c for c in df.columns}
    col = lower_map.get(name.lower())
    if col is None:
        return None
    return df[col]


def _close_column(df: pd.DataFrame) -> pd.Series | None:
    close = _column(df, "Close")
    if close is not None:
        return close
    if df.shape[1] == 1:
        return df.iloc[:, 0]
    return None


def compute_rsi(close: pd.Series, window: int = 14) -> float:
    close = _clean_series(close)
    if len(close) <= window:
        return np.nan
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    latest_loss = avg_loss.iloc[-1]
    latest_gain = avg_gain.iloc[-1]
    if pd.isna(latest_loss) or pd.isna(latest_gain):
        return np.nan
    if latest_loss == 0:
        return 100.0 if latest_gain > 0 else np.nan
    rs = latest_gain / latest_loss
    return float(100 - (100 / (1 + rs)))


def compute_sma(close: pd.Series, window: int) -> float:
    close = _clean_series(close)
    if len(close) < window:
        return np.nan
    return float(close.rolling(window).mean().iloc[-1])


def compute_ema(close: pd.Series, window: int) -> float:
    close = _clean_series(close)
    if len(close) < window:
        return np.nan
    return float(close.ewm(span=window, adjust=False).mean().iloc[-1])


def compute_distance_from_ma(close: pd.Series, window: int) -> float:
    close = _clean_series(close)
    ma = compute_sma(close, window)
    latest_close = _latest(close)
    if pd.isna(ma) or pd.isna(latest_close) or ma == 0:
        return np.nan
    return float((latest_close / ma - 1) * 100)


def compute_drawdown_from_high(close: pd.Series, window: int) -> float:
    close = _clean_series(close)
    if len(close) < 1:
        return np.nan
    recent = close.tail(window)
    rolling_high = recent.max()
    latest_close = recent.iloc[-1]
    if pd.isna(rolling_high) or rolling_high == 0:
        return np.nan
    return float((latest_close / rolling_high - 1) * 100)


def compute_days_since_high(close: pd.Series, window: int) -> float:
    close = _clean_series(close)
    if close.empty:
        return np.nan
    recent = close.tail(window)
    if recent.empty:
        return np.nan
    high_pos = int(np.argmax(recent.to_numpy()))
    return float(len(recent) - 1 - high_pos)


def compute_bollinger_percent_b(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> float:
    close = _clean_series(close)
    if len(close) < window:
        return np.nan
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std()
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    width = upper.iloc[-1] - lower.iloc[-1]
    if pd.isna(width) or width == 0:
        return np.nan
    return float((close.iloc[-1] - lower.iloc[-1]) / width)


def compute_zscore(close: pd.Series, window: int = 20) -> float:
    close = _clean_series(close)
    if len(close) < window:
        return np.nan
    recent = close.tail(window)
    sd = recent.std()
    if pd.isna(sd) or sd == 0:
        return np.nan
    return float((recent.iloc[-1] - recent.mean()) / sd)


def compute_atr_percent(
    high: pd.Series | None,
    low: pd.Series | None,
    close: pd.Series | None,
    window: int = 14,
) -> float:
    high = _clean_series(high)
    low = _clean_series(low)
    close = _clean_series(close)
    if high.empty or low.empty or close.empty:
        return np.nan
    aligned = pd.concat({"high": high, "low": low, "close": close}, axis=1).dropna()
    if len(aligned) <= window:
        return np.nan
    prev_close = aligned["close"].shift(1)
    true_range = pd.concat(
        [
            aligned["high"] - aligned["low"],
            (aligned["high"] - prev_close).abs(),
            (aligned["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(window).mean().iloc[-1]
    latest_close = aligned["close"].iloc[-1]
    if pd.isna(atr) or latest_close == 0:
        return np.nan
    return float((atr / latest_close) * 100)


def compute_volume_ratio(
    volume: pd.Series | None,
    short_window: int = 20,
    long_window: int = 60,
) -> float:
    volume = _clean_series(volume)
    if len(volume) < long_window:
        return np.nan
    long_avg = volume.tail(long_window).mean()
    short_avg = volume.tail(short_window).mean()
    if pd.isna(long_avg) or long_avg == 0:
        return np.nan
    return float(short_avg / long_avg)


def compute_down_day_volume_ratio(
    close: pd.Series,
    volume: pd.Series | None,
    window: int = 20,
) -> float:
    close = _clean_series(close)
    volume = _clean_series(volume)
    if close.empty or volume.empty:
        return np.nan
    aligned = pd.concat({"close": close, "volume": volume}, axis=1).dropna()
    if len(aligned) < window + 1:
        return np.nan
    recent = aligned.tail(window + 1).copy()
    recent["is_down"] = recent["close"].diff() < 0
    window_rows = recent.iloc[-window:]
    all_avg = window_rows["volume"].mean()
    down_avg = window_rows.loc[window_rows["is_down"], "volume"].mean()
    if pd.isna(all_avg) or all_avg == 0 or pd.isna(down_avg):
        return np.nan
    return float(down_avg / all_avg)


def _return_over_window(close: pd.Series, window: int) -> float:
    close = _clean_series(close)
    if len(close) <= window:
        return np.nan
    start = close.iloc[-window - 1]
    end = close.iloc[-1]
    if pd.isna(start) or start == 0:
        return np.nan
    return float((end / start - 1) * 100)


def _realized_vol(close: pd.Series, window: int = 20) -> float:
    close = _clean_series(close)
    rets = close.pct_change().dropna().tail(window)
    if len(rets) < max(2, window // 2):
        return np.nan
    return float(rets.std() * np.sqrt(252) * 100)


def _rsi_percentile_1y(close: pd.Series, window: int = 14) -> float:
    close = _clean_series(close)
    if len(close) < 252 + window:
        return np.nan
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    recent = rsi.dropna().tail(252)
    if recent.empty:
        return np.nan
    latest = recent.iloc[-1]
    return float(recent.rank(pct=True).iloc[-1] * 100) if pd.notna(latest) else np.nan


def compute_technical_metrics(price_history_by_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for ticker, history in (price_history_by_ticker or {}).items():
        if history is None or not isinstance(history, pd.DataFrame) or history.empty:
            continue
        close = _close_column(history)
        if close is None:
            continue
        close = _clean_series(close)
        if close.empty:
            continue

        high = _column(history, "High")
        low = _column(history, "Low")
        volume = _column(history, "Volume")
        sma_50 = compute_sma(close, 50)
        sma_200 = compute_sma(close, 200)

        try:
            rows[str(ticker)] = {
                "RSI_14": compute_rsi(close, 14),
                "RSI_14_Percentile_1Y": _rsi_percentile_1y(close, 14),
                "Distance_50DMA": compute_distance_from_ma(close, 50),
                "Distance_200DMA": compute_distance_from_ma(close, 200),
                "SMA_50_200_Spread": (
                    float((sma_50 / sma_200 - 1) * 100)
                    if pd.notna(sma_50) and pd.notna(sma_200) and sma_200 != 0
                    else np.nan
                ),
                "Drawdown_3M_High": compute_drawdown_from_high(close, 63),
                "Drawdown_1Y_High": compute_drawdown_from_high(close, 252),
                "Days_Since_3M_High": compute_days_since_high(close, 63),
                "Return_20D": _return_over_window(close, 20),
                "Return_60D": _return_over_window(close, 60),
                "Realized_Vol_20D": _realized_vol(close, 20),
                "ATR_14_Pct": compute_atr_percent(high, low, close, 14),
                "Bollinger_PctB_20D": compute_bollinger_percent_b(close, 20, 2.0),
                "ZScore_20D": compute_zscore(close, 20),
                "Volume_Ratio_20D_60D": compute_volume_ratio(volume, 20, 60),
                "Down_Day_Volume_Ratio_20D": compute_down_day_volume_ratio(close, volume, 20),
            }
        except Exception:
            rows[str(ticker)] = {}

    return pd.DataFrame.from_dict(rows, orient="index")


def _pct_rank(series: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    ranks = values.rank(pct=True)
    if not higher_is_better:
        ranks = 1 - ranks
    return ranks


def _bounded_pullback_score(drawdown: pd.Series) -> pd.Series:
    dd = pd.to_numeric(drawdown, errors="coerce")
    magnitude = (-dd).clip(lower=0)
    # Reward normal pullbacks up to roughly 15%, then fade collapses above 30%.
    score = (magnitude / 15).clip(upper=1)
    collapse_penalty = ((magnitude - 30) / 30).clip(lower=0, upper=1)
    return (score * (1 - collapse_penalty)).clip(lower=0, upper=1)


def _rsi_dip_score(rsi: pd.Series) -> pd.Series:
    r = pd.to_numeric(rsi, errors="coerce")
    score = pd.Series(np.nan, index=r.index, dtype=float)
    score = score.mask(r < 25, 0.45)
    score = score.mask((r >= 25) & (r <= 45), 1.0)
    score = score.mask((r > 45) & (r <= 60), (60 - r) / 15)
    score = score.mask(r > 60, 0.0)
    return score


def _bearish_rsi_score(rsi: pd.Series) -> pd.Series:
    r = pd.to_numeric(rsi, errors="coerce")
    score = pd.Series(np.nan, index=r.index, dtype=float)
    score = score.mask(r < 25, 0.35)
    score = score.mask((r >= 25) & (r < 30), 0.65)
    score = score.mask((r >= 30) & (r <= 50), 1.0)
    score = score.mask((r > 50) & (r <= 65), (65 - r) / 15)
    score = score.mask(r > 65, 0.0)
    return score


def _weighted_score(parts: list[tuple[pd.Series, float]], min_weight: float = 0.35) -> pd.Series:
    if not parts:
        return pd.Series(dtype=float)
    index = parts[0][0].index
    numerator = pd.Series(0.0, index=index)
    denominator = pd.Series(0.0, index=index)
    for values, weight in parts:
        values = pd.to_numeric(values, errors="coerce")
        valid = values.notna()
        numerator = numerator.add(values.fillna(0) * weight, fill_value=0)
        denominator = denominator.add(valid.astype(float) * weight, fill_value=0)
    score = numerator / denominator.replace(0, np.nan)
    score = score.where(denominator >= min_weight)
    return (score.clip(0, 1) * 100).round(2)


def compute_signal_scores(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.copy()
    if df.empty:
        return df

    sharpe_col = "Annualized Sharpe" if "Annualized Sharpe" in df.columns else "Sharpe"
    sortino_col = "Sortino Ratio" if "Sortino Ratio" in df.columns else "Sortino"

    dip_parts: list[tuple[pd.Series, float]] = []
    if sharpe_col in df:
        dip_parts.append((_pct_rank(df[sharpe_col]), 0.18))
    if sortino_col in df:
        dip_parts.append((_pct_rank(df[sortino_col]), 0.16))
    if "CAGR" in df:
        dip_parts.append((_pct_rank(df["CAGR"]), 0.14))
    if "Drawdown_3M_High" in df:
        dip_parts.append((_bounded_pullback_score(df["Drawdown_3M_High"]), 0.18))
    if "RSI_14" in df:
        dip_parts.append((_rsi_dip_score(df["RSI_14"]), 0.12))
    if "Distance_200DMA" in df:
        dist = pd.to_numeric(df["Distance_200DMA"], errors="coerce")
        near_trend = ((dist + 10) / 20).clip(lower=0, upper=1)
        dip_parts.append((near_trend, 0.12))
    if "Realized_Vol_20D" in df:
        dip_parts.append((_pct_rank(df["Realized_Vol_20D"], higher_is_better=False), 0.05))
    if "ATR_14_Pct" in df:
        dip_parts.append((_pct_rank(df["ATR_14_Pct"], higher_is_better=False), 0.05))
    dip = _weighted_score(dip_parts)
    if "Distance_200DMA" in df:
        dip = dip.mask(pd.to_numeric(df["Distance_200DMA"], errors="coerce") < -10, dip * 0.65)
    df["Dip_Buy_Score"] = dip.round(2)

    bear_parts: list[tuple[pd.Series, float]] = []
    for col, weight in [
        ("Return_20D", 0.14),
        ("Return_60D", 0.16),
        ("Distance_50DMA", 0.14),
        ("Distance_200DMA", 0.14),
        ("SMA_50_200_Spread", 0.12),
        ("Drawdown_3M_High", 0.10),
    ]:
        if col in df:
            bear_parts.append((_pct_rank(df[col], higher_is_better=False), weight))
    if "ATR_14_Pct" in df:
        bear_parts.append((_pct_rank(df["ATR_14_Pct"]), 0.08))
    if "Down_Day_Volume_Ratio_20D" in df:
        bear_parts.append((_pct_rank(df["Down_Day_Volume_Ratio_20D"]), 0.06))
    if "RSI_14" in df:
        bear_parts.append((_bearish_rsi_score(df["RSI_14"]), 0.06))
    df["Bear_Breakdown_Score"] = _weighted_score(bear_parts)

    momentum_parts: list[tuple[pd.Series, float]] = []
    for col, weight in [
        ("Return_20D", 0.24),
        ("Return_60D", 0.24),
        ("Distance_50DMA", 0.16),
        ("SMA_50_200_Spread", 0.14),
        ("Volume_Ratio_20D_60D", 0.10),
    ]:
        if col in df:
            momentum_parts.append((_pct_rank(df[col]), weight))
    if "Realized_Vol_20D" in df:
        momentum_parts.append((_pct_rank(df["Realized_Vol_20D"], higher_is_better=False), 0.08))
    if "ATR_14_Pct" in df:
        momentum_parts.append((_pct_rank(df["ATR_14_Pct"], higher_is_better=False), 0.04))
    df["Momentum_Continuation_Score"] = _weighted_score(momentum_parts)

    return df
