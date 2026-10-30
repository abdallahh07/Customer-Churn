import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Customer Churn API")

model = joblib.load("models/churn_model.pkl")

FEATURES = [
    "recency", "tenure", "frequency", "monetary", "avg_order_value",
    "recent_orders", "return_rate", "unique_products", "avg_days_between",
]


class CustomerFeatures(BaseModel):
    recency: int
    tenure: int
    frequency: int
    monetary: float
    avg_order_value: float
    recent_orders: int
    return_rate: float
    unique_products: int
    avg_days_between: float


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerFeatures):
    df = pd.DataFrame([customer.model_dump()])[FEATURES]

    prob_return = model.predict_proba(df)[0][1]
    prediction = int(model.predict(df)[0])

    return {
        "prediction": prediction,
        "label": "returned" if prediction == 1 else "churned",
        "churn_probability": round(1 - prob_return, 3),
    }