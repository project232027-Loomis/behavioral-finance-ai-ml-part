# Dataset Info — NIFTY-50 Stock Market Data (2000–2021)

## Source
**Kaggle:** [rohanrao/nifty50-stock-market-data](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data)
**Author:** Vopani (rohanrao)
**License:** CC0: Public Domain
**Provider:** NSE India (National Stock Exchange)

## Description
Day-level price history and trading volumes for all fifty stocks in the NIFTY-50 index, covering **1 January 2000 – 30 April 2021**. Data is split into individual CSV files per stock, plus a combined file and a metadata file.

## Files
- 52 files total (one CSV per stock, e.g. `ADANIPORTS.csv`, `INFY.csv`, `TCS.csv`, etc.)
- `NIFTY50_all.csv` — combined dataset across all stocks
- Metadata file with macro-level stock information

## Columns (per stock file)
| Column | Description |
|---|---|
| Date | Trading date |
| Symbol | Stock ticker symbol |
| Series | Type of security (e.g. EQ) |
| Prev Close | Previous day's closing price |
| Open | Opening price |
| High | Highest price of the day |
| Low | Lowest price of the day |
| Last | Last traded price |
| Close | Closing price |
| VWAP | Volume Weighted Average Price |
| Volume | Number of shares traded |
| Turnover | Total value of shares traded (₹) |
| Trades | Number of trades executed |
| Deliverable Volume | Shares actually transferred (not intraday squared-off) |
| %Deliverble | Deliverable Volume as a % of total Volume |

### Sample Row
```csv
Date,Symbol,Series,Prev Close,Open,High,Low,Last,Close,VWAP,Volume,Turnover,Trades,Deliverable Volume,%Deliverble
2007-11-27,MUNDRAPORT,EQ,440.0,770.0,1050.0,770.0,959.0,962.9,984.72,27294366,2687719053785000.0,,9859619,0.3612
```
> Note: `MUNDRAPORT` was the earlier ticker symbol for what is now `ADANIPORTS` — the company was renamed, so early rows in `ADANIPORTS.csv` carry the old symbol. Also note the `Trades` column is empty for older rows — this field was only introduced by NSE partway through the dataset's date range, so it should be treated as missing (not zero) during preprocessing.

## Update Frequency
Monthly (source dataset). For this project, the data is used as a **static historical snapshot** for training and backtesting — no live refresh is required since the system operates in a simulation environment.

## Usage in This Project
This dataset is used directly (no synthetic substitution) as the historical market data source for:
- **SSI Module** — Multi-Factor Weighted Scoring (Trend/Volatility/Volume) and XGBoost exit-signal training
- **Backtesting** — Walk-forward validation across market cycles (e.g. 2008 crash conditions present in the data range)

## Download
```bash
kaggle datasets download -d rohanrao/nifty50-stock-market-data -p data/ise-data --unzip
```

## Notes
- Data is at day-level granularity only — no intraday/high-frequency data available.
- CC0 license — free to use, modify, and redistribute without attribution requirements (attribution given here as good practice).