import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "Return1",
    "Return5",
    "Return20",
    "RPD10",
    "Volatility20",
    "RelativeVolume",
    "RSI14"
]


def calculate_features(df):

    df = df.copy()

    # Ensure chronological ordering.
    df = df.sort_values("Date").reset_index(drop=True)

    close = df["Adj Close"].astype(float)
    volume = df["Volume"].astype(float)

    # --------------------------------------------------
    # 1. Returns
    # --------------------------------------------------

    df["Return1"] = close.pct_change(1)
    df["Return5"] = close.pct_change(5)
    df["Return20"] = close.pct_change(20)

    # --------------------------------------------------
    # 2. Relative Price Deviation from MA10
    # --------------------------------------------------

    ma10 = close.rolling(window=10).mean()

    df["RPD10"] = (close - ma10) / ma10

    # --------------------------------------------------
    # 3. Volatility
    # --------------------------------------------------

    df["Volatility20"] = (
        df["Return1"]
        .rolling(window=20)
        .std(ddof=0)
    )

    # --------------------------------------------------
    # 4. Relative Volume
    # --------------------------------------------------

    # Average volume of the PREVIOUS 20 sessions.
    avg_volume20 = (
        volume
        .rolling(window=20)
        .mean()
        .shift(1)
    )

    df["RelativeVolume"] = volume / avg_volume20

    # --------------------------------------------------
    # 5. RSI (Wilder's smoothing)
    # --------------------------------------------------

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    period = 14

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    # Handle edge cases.
    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain > 0),
        100
    )

    rsi = rsi.mask(
        (avg_gain == 0) & (avg_loss > 0),
        0
    )

    rsi = rsi.mask(
        (avg_gain == 0) & (avg_loss == 0),
        50
    )

    df["RSI14"] = rsi

    # --------------------------------------------------
    # 6. Clean invalid values
    # --------------------------------------------------

    df = df.replace([np.inf, -np.inf], np.nan)

    # Remove rows where any feature is unavailable.
    df = df.dropna(subset=FEATURE_COLUMNS)

    df = df.reset_index(drop=True)

    return df