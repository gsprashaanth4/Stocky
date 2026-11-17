import yfinance as yf
import pandas as pd
import numpy as np
import os


STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "AMD", "AVGO", "ORCL", "CSCO",
    "JPM", "BAC", "GS", "V", "MA",
    "XOM", "CVX",
    "CAT", "HON", "GE",
    "WMT", "COST", "KO", "PG", "MCD",
    "JNJ", "ABBV", "UNH",
    "DIS", "UBER"
]


REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits"
]


def download_stock(ticker, start="2000-01-01", end=None):

    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=True,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    # Handle yfinance MultiIndex columns.
    if isinstance(df.columns, pd.MultiIndex):

        # Single-ticker download: remove the ticker level.
        for level in range(df.columns.nlevels):
            if ticker in df.columns.get_level_values(level):
                df.columns = df.columns.droplevel(level)
                break

    # Ensure all expected columns exist.
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            if col in ["Dividends", "Stock Splits"]:
                df[col] = 0.0
            else:
                raise ValueError(
                    f"{ticker}: missing required column {col}"
                )

    df = df[REQUIRED_COLUMNS].copy()

    # Keep the date as an explicit column.
    df.index.name = "Date"
    df = df.reset_index()

    df["Ticker"] = ticker

    # Basic data validation.
    df = df.dropna(
        subset=["Open", "High", "Low", "Close", "Volume"]
    )

    df = df.sort_values("Date").reset_index(drop=True)

    if (df["Close"] <= 0).any():
        raise ValueError(f"{ticker}: non-positive closing price")

    if (df["Volume"] < 0).any():
        raise ValueError(f"{ticker}: negative volume")

    return df


def download_universe(
    stocks=STOCKS,
    start="2000-01-01",
    end=None,
    save_dir="data"
):

    os.makedirs(save_dir, exist_ok=True)

    datasets = {}

    for ticker in stocks:

        print(f"Downloading {ticker} ...")

        try:
            df = download_stock(ticker, start, end)

            datasets[ticker] = df

            path = os.path.join(save_dir, f"{ticker}.csv")
            df.to_csv(path, index=False)

            print(
                f"  {len(df)} rows | "
                f"{df['Date'].min().date()} -> "
                f"{df['Date'].max().date()}"
            )

        except Exception as e:
            print(f"  FAILED: {ticker}: {e}")

    return datasets