from typing import Dict, Tuple

# --- Order-agnostic pair handling helpers -----------------------------------
# Declare which authored pairs can be safely mirrored when X/Y are swapped.
SYMMETRIC_PAIRS = {
    ("Mean Daily Return", "Daily Volatility (Std)"),
    ("Annualized Sharpe", "Max Drawdown"),
    ("Sortino Ratio", "Downside Deviation"),
    ("RSI(14)", "Bollinger Bandwidth (20)"),
    ("% Above 50DMA", "Bollinger Bandwidth (20)"),
    ("% Above 200DMA", "Mean Daily Return"),
    ("Return Skewness", "Return Kurtosis (Fisher)"),
}

# Flip helper: turn "X • Y → ..." into "Y • X → ..." while preserving the tail text.
def _flip_bullet(s: str) -> str:
    parts = s.split(" • ", 1)
    if len(parts) == 2:
        # Keep anything after the arrow as-is.
        left, rest = parts[0], parts[1]
        # If there is an arrow later in the string, preserve it untouched.
        # We only swap the left-hand bullet order.
        return f"{rest.split(' → ', 1)[0]} • {left}{' → ' + rest.split(' → ', 1)[1] if ' → ' in rest else ''}"
    return s

# Apply flip to all quadrant strings of a guide dict.
def _flip_quadrants_text(qdict: Dict[str, str]) -> Dict[str, str]:
    return {k: _flip_bullet(v) for k, v in qdict.items()}

# Utility: check set membership regardless of order
def _unordered_in(pair: tuple) -> bool:
    a = tuple(sorted(pair))
    return any(a == tuple(sorted(p)) for p in SYMMETRIC_PAIRS)

# --- Metric glossary (aligned with UI) --------------------------------------

METRIC_META: Dict[str, Dict[str, str]] = {
    "Mean Daily Return": {
        "role": "return",
        "desc": "Average daily arithmetic return over the lookback window.",
        "insight": "Higher is better; judge quality using risk and drawdown metrics.",
    },
    "Daily Volatility (Std)": {
        "role": "risk_vol",
        "desc": "Standard deviation of daily returns.",
        "insight": "Measures path choppiness; not all high-vol assets reward the risk.",
    },
    "Annualized Sharpe": {
        "role": "risk_adj",
        "desc": "Risk-adjusted return (mean/vol) scaled to annual units.",
        "insight": "Combines reward and risk efficiency; higher = better.",
    },
    "Max Drawdown": {
        "role": "risk_path",
        "desc": "Worst peak-to-trough loss over the lookback period.",
        "insight": "Measures pain/ruin risk; lower magnitude = better.",
    },
    "CAGR": {
        "role": "compounding",
        "desc": "Annualized geometric growth rate over the lookback.",
        "insight": "Long-term growth perspective; smooth compounding beats sporadic spikes.",
    },
    "Downside Deviation": {
        "role": "risk_down",
        "desc": "Standard deviation of negative daily returns.",
        "insight": "Focuses on downside volatility; smaller = more stable downside.",
    },
    "Sortino Ratio": {
        "role": "risk_adj_down",
        "desc": "Return / downside deviation (annualized).",
        "insight": "Reward per unit of bad-vol; higher is better.",
    },
    "RSI(14)": {
        "role": "oscillator",
        "desc": "Overbought/oversold oscillator (0–100).",
        "insight": "High = overbought risk; low = oversold risk.",
    },
    "% Above 50DMA": {
        "role": "trend_level_50",
        "desc": "Percent distance from the 50-day moving average.",
        "insight": "Above 0 = above trend; larger = more stretched.",
    },
    "% Above 200DMA": {
        "role": "trend_level_200",
        "desc": "Percent distance from the 200-day moving average.",
        "insight": "Measures long-term trend strength; >0% = bullish regime.",
    },
    "Return Skewness": {
        "role": "shape_skew",
        "desc": "Third standardized moment of returns (asymmetry).",
        "insight": "Positive = upside tail bias; negative = crash risk.",
    },
    "Return Kurtosis (Fisher)": {
        "role": "shape_kurt",
        "desc": "Excess kurtosis (0 = normal).",
        "insight": "Higher = fatter tails, more extreme moves.",
    },
    "Bollinger Bandwidth (20)": {
        "role": "range_vol",
        "desc": "Relative band width (Upper−Lower)/Middle.",
        "insight": "Low BW = squeeze; high BW = expansion/trend.",
    },
}

# --- Quadrant guides --------------------------------------------------------

PAIR_GUIDES: Dict[Tuple[str, str], Dict[str, str]] = {
    ("Mean Daily Return", "Daily Volatility (Std)"): {
        "TR": "High return • High vol → Hot but wild (size down).",
        "TL": "Low return • High vol → Poor payoff to risk.",
        "BR": "High return • Low vol → Quality compounding (best).",
        "BL": "Low return • Low vol → Quiet laggard / dead money.",
    },
    ("Annualized Sharpe", "Max Drawdown"): {
        "TR": "High Sharpe • High DD → Efficient but rough path.",
        "TL": "Low Sharpe • High DD → Worst quadrant (avoid).",
        "BR": "High Sharpe • Low DD → Elite performers.",
        "BL": "Low Sharpe • Low DD → Stable but dull.",
    },
    ("Sortino Ratio", "Downside Deviation"): {
        "TR": "High Sortino • High downside dev → Strong reward despite crashes.",
        "TL": "Low Sortino • High downside dev → Poor payoff to pain.",
        "BR": "High Sortino • Low downside dev → Excellent quality of returns.",
        "BL": "Low Sortino • Low downside dev → Calm but underperforming.",
    },
    ("RSI(14)", "Bollinger Bandwidth (20)"): {
        "TR": "High RSI • High BW → Overbought with expansion (blow-off risk).",
        "TL": "Low RSI • High BW → Oversold with expansion (capitulation risk).",
        "BR": "High RSI • Low BW → Tight overbought coil (watch fade).",
        "BL": "Low RSI • Low BW → Quiet oversold coil (watch bounce).",
    },
    ("% Above 200DMA", "Mean Daily Return"): {
        "TR": "Above long MA • High return → Strong trend leader.",
        "TL": "Below long MA • High return → Early reversal candidate.",
        "BR": "Above long MA • Low return → Mature uptrend pausing.",
        "BL": "Below long MA • Low return → Structural laggard.",
    },
    ("% Above 50DMA", "Bollinger Bandwidth (20)"): {
        "TR": "Above 50DMA • High BW → Trend in expansion.",
        "TL": "Below 50DMA • High BW → Volatile down move.",
        "BR": "Above 50DMA • Low BW → Calm consolidation.",
        "BL": "Below 50DMA • Low BW → Quiet base-building.",
    },
    ("Return Skewness", "Return Kurtosis (Fisher)"): {
        "TR": "Right-tail + Fat tails → Explosive upside (and risk).",
        "TL": "Left-tail + Fat tails → Crash-prone.",
        "BR": "Right-tail + Thin tails → Gentle upside bias.",
        "BL": "Left-tail + Thin tails → Grind-down regime.",
    },
}

# --- Role-based fallback guides --------------------------------------------

ROLE_READS: Dict[Tuple[str, str], Dict[str, str]] = {
    ("return", "risk_vol"): {
        "TR": "High return • High vol → Hot but wild.",
        "TL": "Low return • High vol → Poor payoff.",
        "BR": "High return • Low vol → Efficient compounder.",
        "BL": "Low return • Low vol → Laggard.",
    },
    ("risk_adj", "risk_path"): {
        "TR": "Efficient returns but drawdown heavy.",
        "TL": "Inefficient and drawdown heavy.",
        "BR": "Efficient and smooth path (ideal).",
        "BL": "Stable but uninspired.",
    },
    ("risk_adj_down", "risk_down"): {
        "TR": "High Sortino despite high downside risk.",
        "TL": "Low Sortino with high downside risk.",
        "BR": "High Sortino • Low downside → Elite profile.",
        "BL": "Low Sortino • Low downside → Conservative laggard.",
    },
    ("oscillator", "range_vol"): {
        "TR": "Overbought with expansion → Blow-off risk.",
        "TL": "Oversold with expansion → Capitulation.",
        "BR": "Overbought with squeeze → Potential fade.",
        "BL": "Oversold with squeeze → Potential rebound.",
    },
    ("trend_level_200", "return"): {
        "TR": "Above MA • High return → Trend leader.",
        "TL": "Below MA • High return → Reversal candidate.",
        "BR": "Above MA • Low return → Mature trend.",
        "BL": "Below MA • Low return → Laggard.",
    },
    ("trend_level_50", "range_vol"): {
        "TR": "Above 50DMA • Expansion → Trend continuation.",
        "TL": "Below 50DMA • Expansion → Downtrend continuation.",
        "BR": "Above 50DMA • Squeeze → Calm consolidation.",
        "BL": "Below 50DMA • Squeeze → Base formation.",
    },
    ("shape_skew", "shape_kurt"): {
        "TR": "Right-tail with fat tails → Explosive upside & risk.",
        "TL": "Left-tail with fat tails → Crash-prone.",
        "BR": "Right-tail with thin tails → Balanced upside bias.",
        "BL": "Left-tail with thin tails → Gradual decay.",
    },
}

# --- Helper functions (unchanged) ------------------------------------------

def _role_of(metric: str) -> str:
    meta = METRIC_META.get(metric, {})
    return meta.get("role", "")

def get_pair_guide(x_metric: str, y_metric: str) -> Dict[str, object]:
    # 1) Exact authored (x, y)
    guide = PAIR_GUIDES.get((x_metric, y_metric))

    # 2) Authored in reversed order (y, x) and the pair is symmetric → flip text
    if guide is None:
        rev_key = (y_metric, x_metric)
        if rev_key in PAIR_GUIDES and _unordered_in(rev_key):
            guide = _flip_quadrants_text(PAIR_GUIDES[rev_key])

    # 3) Role-based fallback
    if guide is None:
        x_role, y_role = _role_of(x_metric), _role_of(y_metric)
        guide = ROLE_READS.get((x_role, y_role))

    # 4) Generic fallback
    if guide is None:
        guide = {
            "TR": f"High {x_metric} • High {y_metric} → Upper-right quadrant.",
            "TL": f"Low {x_metric} • High {y_metric} → Upper-left quadrant.",
            "BR": f"High {x_metric} • Low {y_metric} → Lower-right quadrant.",
            "BL": f"Low {x_metric} • Low {y_metric} → Lower-left quadrant.",
        }

    def _meta(m: str) -> Dict[str, str]:
        mm = METRIC_META.get(m, {})
        return {
            "metric": m,
            "role": mm.get("role", ""),
            "desc": mm.get("desc", ""),
            "insight": mm.get("insight", ""),
        }

    return {"x_meta": _meta(x_metric), "y_meta": _meta(y_metric), "quadrants": guide}