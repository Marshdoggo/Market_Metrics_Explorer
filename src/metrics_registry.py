import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from utils import to_annualized, pct_from_ma

# All metric functions accept a single-column price series for one ticker
# and return a scalar float.
def mean_daily_return(prices: pd.Series) -> float:
    rets = prices.pct_change().dropna()
    return float(rets.mean())

def daily_volatility(prices: pd.Series) -> float:
    rets = prices.pct_change().dropna()
    return float(rets.std())

def annualized_sharpe(prices: pd.Series) -> float:
    rets = prices.pct_change().dropna()
    return float(to_annualized(rets.mean(), rets.std()))

def max_drawdown(prices: pd.Series) -> float:
    roll_max = prices.cummax()
    dd = prices/roll_max - 1.0
    return float(dd.min())

def cagr(prices: pd.Series, periods_per_year=252) -> float:
    if len(prices) < 2:
        return np.nan
    total_return = prices.iloc[-1] / prices.iloc[0]
    years = len(prices) / periods_per_year
    if years <= 0 or total_return <= 0:
        return np.nan
    return float(total_return ** (1/years) - 1)

def downside_deviation(prices: pd.Series, mar=0.0) -> float:
    rets = prices.pct_change().dropna()
    downside = np.minimum(rets - mar/252.0, 0)
    return float(np.sqrt(np.mean(downside**2)))

def sortino(prices: pd.Series) -> float:
    rets = prices.pct_change().dropna()
    dd = downside_deviation(prices)
    if dd == 0:
        return np.nan
    ann_ret = rets.mean() * 252
    return float(ann_ret / (dd * np.sqrt(252)))

def rsi_14(prices: pd.Series) -> float:
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up.iloc[-1] / roll_down.iloc[-1] if roll_down.iloc[-1] != 0 else np.nan
    return float(100 - (100 / (1 + rs))) if rs==rs else np.nan

def pct_above_50dma(prices: pd.Series) -> float:
    return float(pct_from_ma(prices, 50))

def pct_above_200dma(prices: pd.Series) -> float:
    return float(pct_from_ma(prices, 200))

def return_skewness(prices: pd.Series) -> float:
    rets = prices.pct_change().dropna()
    return float(skew(rets)) if len(rets)>3 else np.nan

def return_kurtosis(prices: pd.Series) -> float:
    rets = prices.pct_change().dropna()
    return float(kurtosis(rets, fisher=True)) if len(rets)>3 else np.nan

def bollinger_bandwidth_20(prices: pd.Series) -> float:
    ma = prices.rolling(20).mean()
    sd = prices.rolling(20).std()
    upper = ma + 2*sd
    lower = ma - 2*sd
    bw = (upper - lower) / ma
    return float(bw.iloc[-1]) if len(bw.dropna()) else np.nan

METRICS = {
    "Mean Daily Return": mean_daily_return,
    "Daily Volatility (Std)": daily_volatility,
    "Annualized Sharpe": annualized_sharpe,
    "Max Drawdown": max_drawdown,
    "CAGR": cagr,
    "Downside Deviation": downside_deviation,
    "Sortino Ratio": sortino,
    "RSI(14)": rsi_14,
    "% Above 50DMA": pct_above_50dma,
    "% Above 200DMA": pct_above_200dma,
    "Return Skewness": return_skewness,
    "Return Kurtosis (Fisher)": return_kurtosis,
    "Bollinger Bandwidth (20)": bollinger_bandwidth_20,
}
