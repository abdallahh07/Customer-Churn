# Customer Churn Prediction: Online Retail

> **Which customers are about to stop buying, and what can the business do about it?**

This project predicts which customers of a UK online gift retailer are likely to stop buying in the next 90 days, so the business can act **before** they leave.

**Contents:** [Part 1: Overview (non-technical)](#part-1-overview) · [Part 2: Technical details](#part-2-technical-details)

---

# Part 1: Overview

## The business problem

Keeping an existing customer costs far less than winning a new one. But online customers rarely announce that they are leaving. They simply stop ordering.

This project answers two questions:

1. **Who** is likely to stop buying in the next 90 days?
2. **Why**: what behaviour signals that a customer is drifting away?

## The data

Real transactions from a UK-based online retailer selling gift-ware, many of them to wholesalers:

- **~1.07 million transactions**, December 2009 to December 2011
- Customers in **40+ countries**
- Source: [UCI Machine Learning Repository, Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

The raw data is messy, as real data usually is: missing customer IDs, duplicates, cancelled orders, returns and accounting adjustments. These are cleaned and documented.

## How "churn" is defined

The retailer has no "cancel" button, so churn has to be defined from behaviour:

> A customer **churns** if they bought before **9 September 2011** but placed **no order in the following 90 days** (9 Sep – 9 Dec 2011).

The model only sees each customer's history **before** that date, exactly as it would in real life.

## Key findings

- **Recency is the strongest signal.** How long ago a customer last bought drives over half of the model's decisions. Customers who have gone quiet rarely come back.
- **Spending and order frequency come next.** Customers who order often and spend more are much more likely to return.
- **Sales are highly seasonal.** Revenue and active customers peak sharply every November (Christmas buying). Some customers buy only around Christmas and then disappear for a year.
- **Revenue comes from more customers, not bigger orders.** Monthly revenue and monthly active customers move together.
- **The UK dominates** the customer base, and the store barely trades on Saturdays.

## Results

Out of **5,341 customers**, **56% churned** in the 90-day window.

| What | Result |
|---|---|
| Accuracy | **74.5%** (vs. 56% from always guessing "churn") |
| How well the model ranks customers by churn risk (ROC-AUC) | **0.82** |
| Share of actual churners the model catches | **83%** (500 of 603 in the test set) |

## What the business can do with this

- **Score every customer** and send a retention offer to those with the highest churn risk.
- **Watch recency closely:** a customer who hasn't ordered for several weeks longer than their usual rhythm is a warning sign.
- **Prioritise high-value wholesale accounts**, where losing one customer costs the most.
- **Plan around seasonality:** re-engage Christmas-only buyers before the next November peak.

---

# Part 2: Technical details

## Pipeline overview

**raw CSV → cleaning → churn labels + customer features (before cutoff) → train/test split → model comparison → tuning → evaluation → saved model → FastAPI + Docker**

## Dataset

| Column | Description |
|---|---|
| `Invoice` | Invoice number; starts with `C` for cancellations |
| `StockCode` | Product code |
| `Description` | Product name |
| `Quantity` | Units per line (negative for returns or cancellations) |
| `InvoiceDate` | Date and time of the transaction |
| `Price` | Unit price in GBP |
| `Customer ID` | Customer identifier (missing for some rows) |
| `Country` | Customer's country |

Period: 2009-12-01 to 2011-12-09.

## Data cleaning

| Step | Rule | Reason |
|---|---|---|
| Missing values | Drop rows with missing `Customer ID` or `Description` | Churn is a per-customer problem; anonymous rows can't be used |
| Duplicates | Drop exact duplicate rows | Double-recorded entries would inflate counts and spending |
| Invalid prices | Drop `Price <= 0` | Stock write-offs and accounting adjustments, not real sales |
| Dates | Convert `InvoiceDate` to datetime | Needed for the time-based cutoff |

**1,067,371 → 797,815 rows** after cleaning.

Returns (negative `Quantity` with a positive price) are **kept**: they are real customer behaviour.

**Outliers:** the IQR rule flagged 68k prices and 116k quantities, but these were mostly normal wholesale orders in a heavily skewed distribution, so they were kept. Extreme values were inspected individually instead (for example, an 80,995-unit order that was cancelled minutes later).

## Churn label

- **Cutoff date:** 2011-09-09
- **Features:** computed only from transactions **before** the cutoff
- **Label window:** the 90 days after the cutoff
- Target column `returned`: **1** = bought again in the window, **0** = churned

| Class | Customers | Share |
|---|---|---|
| Churned (0) | 3,013 | 56% |
| Returned (1) | 2,328 | 44% |

The classes are reasonably balanced, so accuracy is reported alongside ROC-AUC, always compared with the 56% majority baseline.

## Features

All features are computed per customer, only from data **before** the cutoff.

| Feature | Description |
|---|---|
| `recency` | Days since the last purchase |
| `tenure` | Days since the first purchase |
| `frequency` | Number of unique invoices |
| `monetary` | Total spend (net of returns) |
| `avg_order_value` | Monetary ÷ frequency |
| `recent_orders` | Orders in the 90 days before the cutoff |
| `return_rate` | Share of invoices that were cancellations |
| `unique_products` | Number of distinct products bought |
| `avg_days_between` | Average days between orders |

## Modelling

- Customers split **80/20** into train and test (stratified, `random_state=42`)
- All models wrapped in a scikit-learn `Pipeline` with `StandardScaler`
- Models compared: Logistic Regression, Random Forest, Gradient Boosting, LightGBM, XGBoost
- Best model tuned with `GridSearchCV` (54 combinations, 5-fold CV, scored on ROC-AUC)

## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| **Gradient Boosting** | **0.748** | **0.822** |
| LightGBM | 0.728 | 0.812 |
| Logistic Regression | 0.736 | 0.806 |
| Random Forest | 0.733 | 0.797 |
| XGBoost | 0.718 | 0.794 |
| *Majority-class baseline* | *0.564* | *0.500* |

- All models score within ~0.03 ROC-AUC of each other, so performance is limited by the available features rather than by model choice.
- **Tuning** gave no meaningful gain (0.821 ROC-AUC tuned vs. 0.822 default). Cross-validation (0.809) and test (0.821) scores are close, so the model is not overfitting.
- The tuned Gradient Boosting model (`learning_rate=0.01, max_depth=3, n_estimators=300, subsample=0.8`) is saved to `models/churn_model.pkl`.

### Feature importance

| Feature | Importance |
|---|---|
| recency | 0.54 |
| monetary | 0.15 |
| frequency | 0.10 |
| avg_days_between | 0.09 |
| unique_products | 0.07 |
| recent_orders | 0.02 |
| tenure | 0.02 |
| avg_order_value | 0.02 |
| return_rate | 0.00 |

### Confusion matrix (test set)

| | Predicted churned | Predicted returned |
|---|---|---|
| **Actually churned** | 500 | 103 |
| **Actually returned** | 170 | 296 |

- **83% recall on churners:** the model finds most at-risk customers.
- The 170 false alarms (loyal customers flagged as at risk) are the cheaper mistake: at worst they receive an unnecessary offer.

## Project structure

```
customer-churn/
├── data_processing/      # load_data.py, preprocess.py, feature_engineering.py
├── model/                # train.py, evaluate.py, predict.py
├── pipeline/             # training_pipeline.py
├── utils/                # logging_setup.py, exception_handling.py
├── models/               # churn_model.pkl
├── notebooks/            # 1. exploratory analysis, 2. modelling
├── static/               # index.html (web UI)
├── data/raw/             # raw data (not committed)
├── app.py                # FastAPI app
├── Dockerfile
├── requirements.txt
└── README.md
```

## How to run

```bash
git clone https://github.com/abdallahh07/customer-churn.git
cd customer-churn

conda create -n customer-churn python=3.13 -y
conda activate customer-churn
pip install -r requirements.txt
```

Download `online_retail_II.csv` from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and place it in `data/raw/`.

**Train the model:**

```bash
python -m pipeline.training_pipeline
```

**Run the API and web UI:**

```bash
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` for the UI, or `http://127.0.0.1:8000/docs` for the API docs.

**Run with Docker:**

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

### API example

`POST /predict`

```json
{
  "recency": 200, "tenure": 400, "frequency": 2, "monetary": 300,
  "avg_order_value": 150, "recent_orders": 0, "return_rate": 0,
  "unique_products": 10, "avg_days_between": 100
}
```

Response:

```json
{ "prediction": 0, "label": "churned", "churn_probability": 0.9 }
```

## Limitations and next steps

- **Single cutoff:** the model is evaluated on one 90-day window. An out-of-time test (training on an earlier cutoff, testing on a later one) would show how well it holds up over time.
- **Seasonality:** the label window includes the Christmas peak, which may make churn look lower than in other quarters.
- **Non-product codes** (postage, manual charges) are still in the spending figures and should be removed.
- **Limited data:** two years of transactions from one retailer, with no marketing, website or customer-service data.
- **Next:** remove non-product codes, run the out-of-time validation, estimate revenue at risk, and deploy the Docker image to AWS.

---

## Author

**Abdallah Hashad**: junior data scientist with a background in law and finance.
[LinkedIn](https://www.linkedin.com/in/abdallah-hashad-1a6254249) · [GitHub](https://github.com/abdallahh07)
