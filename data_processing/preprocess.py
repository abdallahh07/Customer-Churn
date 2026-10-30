import pandas as pd


def clean_data(data):
    """Drops missing values, duplicates and non-positive prices; parses dates."""
    df = data.dropna()
    df = df.drop_duplicates()
    df = df[df["Price"] > 0].copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df
