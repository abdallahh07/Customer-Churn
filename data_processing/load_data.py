import pandas as pd

def load_data(file_path):
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)

def clean_data(data):
    df = data.dropna()
    df = df.drop_duplicates()
    df = df[df["Price"] > 0]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df