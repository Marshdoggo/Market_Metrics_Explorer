# Market_Metrics_Explorer_
A Modular, interactive market research tool. Plot S&amp;P, Nasdaq, Dow, and Forex pairs across different X,Y axis metrics for visual insights.

<img width="2192" height="1136" alt="MMEScreenshot" src="https://github.com/user-attachments/assets/2f85205c-89da-426e-9f62-1c0b2fbb734b" />


## Overview

Market Metrics Explorer was created as an independent research project to make financial datasets more intuitive and explorable.

Rather than viewing markets through a single chart, the platform allows users to compare hundreds of securities simultaneously across multiple statistical dimensions, helping identify outliers, clusters, sector behavior, and changing market regimes.

Current supported universes include:

- S&P 500

- Nasdaq 100

- Dow Jones Industrial Average

- Major FX currency pairs

---

## Key Features

- Interactive Plotly scatterplots

- Cross-asset metric comparison

- Adjustable lookback windows

- Sector-based visualization

- Time-series comparison tools

- Financial metric documentation

- Market regime exploration

- Integrated AI-assisted chart interpretation

- Technical signal metrics for dip-buy, bearish breakdown, and momentum-continuation research

---

## Example Questions

The platform is designed to help answer questions such as:

- Which stocks have delivered the highest return per unit of volatility?

- How do sectors cluster during different market environments?

- Which assets exhibit unusual behavior relative to peers?

- How does a company's position change when the lookback period changes?

- What relationships exist between return, drawdown, volatility, and Sharpe ratio?

- Which assets are pulled back from recent highs while still holding longer-term trend support?

- Which assets show weakening mid-term momentum and trend breakdown risk?

- Which assets show persistent upside momentum with volume confirmation?

---

## Technical Signal Metrics

Market Metrics Explorer includes an exploratory technical layer for swing-trading research. These indicators and scores are not financial advice and should be treated as universe-relative screening tools, not absolute predictions.

- `Dip_Buy_Score` attempts to identify assets with strong prior/risk-adjusted performance that are currently pulled back but not structurally broken.

- `Bear_Breakdown_Score` attempts to identify assets with deteriorating trend, momentum, and selling-pressure conditions.

- `Momentum_Continuation_Score` attempts to identify assets with persistent upside trend and momentum.

Scores are scaled from 0 to 100 using cross-sectional percentile ranks inside the selected universe. Close-only datasets populate close-based metrics; high/low/volume metrics such as ATR percent and volume ratios are left blank when those fields are unavailable.

---

## Technology Stack

- Python

- Streamlit

- Plotly

- Pandas

- NumPy

- SQLite

- GitHub Actions

- Financial data APIs

---

## Why I Built This

As an independent student of financial markets, macroeconomics, and quantitative analysis, I wanted a tool that could quickly surface patterns that are difficult to see using traditional price charts alone.

The project serves as both a research environment and an exploration of data visualization techniques for financial decision-making.

---
