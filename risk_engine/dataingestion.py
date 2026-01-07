import yfinance as yf
import pandas as pd
from datetime import datetime


def download_nifty_data(start_date="2010-01-01"):
    """
    Downloads NIFTY 50 historical daily data.
    """
    ticker = "^NSEI"
    end_date = datetime.today().strftime("%Y-%m-%d")

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False
    )

    if data.empty:
        raise ValueError("Data download failed.")

    data.reset_index(inplace=True)
    data.columns = [
    c[0].lower().replace(" ", "_") if isinstance(c, tuple)
    else c.lower().replace(" ", "_")
    for c in data.columns
]


    data = data[["date", "open", "high", "low", "close", "volume"]]
    data.dropna(inplace=True)

    return data


if __name__ == "__main__":
    df = download_nifty_data()
    df.to_csv("data/nifty50_daily.csv", index=False)
    print("NIFTY 50 data saved successfully.")
