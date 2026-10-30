import pandas as pd 

def build_labels(data,cutoff):
    before_cutoff = data[data["InvoiceDate"] < cutoff]
    after_cut_off = data[data["InvoiceDate"] > cutoff]

    before_customer = before_cutoff["Customer ID"].unique()
    after_customer = after_cut_off["Customer ID"].unique()
    
    label = pd.DataFrame({"Customer ID":before_customer})
    label["returned"] = label["Customer ID"].isin(after_customer).astype(int)
    return label
  
def build_features(data, cutoff):
    before_cutoff = data[data["InvoiceDate"] < cutoff]

    first_purchase = before_cutoff.groupby("Customer ID")["InvoiceDate"].min()
    last_purchase = before_cutoff.groupby("Customer ID")["InvoiceDate"].max()

    before_cutoff = before_cutoff.copy()
    before_cutoff["revenue"] = before_cutoff["Quantity"] * before_cutoff["Price"]

    features = pd.DataFrame()
    features["recency"] = (cutoff - last_purchase).dt.days
    features["tenure"] = (cutoff - first_purchase).dt.days
    features["frequency"] = before_cutoff.groupby("Customer ID")["Invoice"].nunique()
    features["monetary"] = before_cutoff.groupby("Customer ID")["revenue"].sum()

    features["avg_order_value"] = features["monetary"] / features["frequency"]

    recent = before_cutoff[before_cutoff["InvoiceDate"] >= cutoff - pd.Timedelta(days=90)]
    features["recent_orders"] = recent.groupby("Customer ID")["Invoice"].nunique()
    features["recent_orders"] = features["recent_orders"].fillna(0)

    is_cancel = before_cutoff["Invoice"].astype(str).str.startswith("C")
    cancels = before_cutoff[is_cancel].groupby("Customer ID")["Invoice"].nunique()
    features["return_rate"] = (cancels / features["frequency"]).fillna(0)

    features["unique_products"] = before_cutoff.groupby("Customer ID")["StockCode"].nunique()

    span = (last_purchase - first_purchase).dt.days
    features["avg_days_between"] = (span / (features["frequency"] - 1)).fillna(0)

    features = features.reset_index()
    return features
  
def merge(lables,features):
    data = features.merge(lables,on = "Customer ID")
    return data
  
def split_xy(dataset):
    x = dataset.drop(columns=["returned", "Customer ID"])    
    y = dataset["returned"]
    return x, y