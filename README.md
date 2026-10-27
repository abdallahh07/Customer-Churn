# Customer Churn Prediction: Online Retail

> **Which customers are about to stop buying, and what can the business do about it?**

This project predicts which customers of a UK online gift retailer are likely to stop buying in the next 90 days, so the business can act **before** they leave.

**Contents:** [Part 1: Overview (non-technical)](#part-1-overview) · [Part 2: Technical details](#part-2-technical-details)

---

# Part 1: Overview

## The business problem

Keeping an existing customer costs far less than winning a new one. But online customers rarely announce that they are leaving. They simply stop ordering.

This project answers three questions:

1. **Who** is likely to stop buying in the next 90 days?
2. **Why**: what behaviour signals that a customer is drifting away?
3. **How much** revenue is at risk if they leave?

## The data

Real transactions from a UK-based online retailer selling gift-ware, many of them to wholesalers:

- **~1 million transactions**, December 2009 to December 2011
- Customers in **40+ countries**
- Source: [UCI Machine Learning Repository, Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

The raw data is messy, as real data usually is: missing customer IDs, cancelled orders, returns and non-product charges. All of these are cleaned and documented.

## How "churn" is defined

The retailer has no "cancel" button, so churn has to be defined from behaviour:

> A customer **churns** if they bought before a given date but placed **no order in the following 90 days**.

The model only sees each customer's history **before** that date, exactly as it would in real life.

## Key findings

_To be completed when the analysis is finished._

- Finding 1: …
- Finding 2: …
- Finding 3: …

## Results

_To be completed after modelling._

| What | Result |
|---|---|
| How well the model ranks customers by churn risk (ROC-AUC) | … |
| Share of churners found in the top 20% riskiest customers | … |
| Revenue at risk from predicted churners | … |

## What the business can do with this

_To be completed. Example: target the top-risk customers with a personalised offer before they lapse, and focus on high-value wholesale accounts first._

---

# Part 2: Technical details

## Pipeline overview

End-to-end churn prediction on real e-commerce transactions:

**raw CSV → cleaning → SQLite → SQL feature engineering → time-based split → model comparison → evaluation**

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

Period: 2009-12-01 to 2011-12-09. Source: UCI Machine Learning Repository, Online Retail II.

## Data cleaning

_Fill in with the final decisions and row counts at each step._

| Step | Rule | Rows removed |
|---|---|---|
| Missing customers | Drop rows without `Customer ID` | … |
| Cancellations | Handle invoices starting with `C` | … |
| Invalid values | Drop `Quantity <= 0` or `Price <= 0` | … |
| Non-product codes | Drop postage, fees, adjustments (e.g. `POST`, `M`, `BANK CHARGES`) | … |
| Duplicates | Drop exact duplicate rows | … |

The cleaned data is stored in a SQLite table (`transactions`) so features can be built with SQL.

## Churn label

- **Cutoff date:** features use only transactions **before** the cutoff.
- **Label window:** 90 days after the cutoff.
- `churn = 1` if a customer with at least one purchase before the cutoff makes **no purchase** in the window.

| Split | Feature period | Label window |
|---|---|---|
| Train | before 2011-06-01 | 2011-06-01 to 2011-09-01 |
| Test | before 2011-09-01 | 2011-09-01 to 2011-12-01 |

A **time-based split** is used instead of a random split, so the model is always evaluated on a later period than it was trained on. This mirrors real deployment and prevents leakage from the future.

## Features

_Update with the final feature list._

| Group | Features |
|---|---|
| RFM | `recency_days`, `frequency`, `monetary`, `avg_order_value`, `tenure_days`, `orders_per_month` |
| Behaviour | number of distinct products, basket size, return/cancellation rate |
| Profile | country |

All features are computed only from data **before** the cutoff.

## Modelling

- Baseline: Logistic Regression (scaled, inside a scikit-learn `Pipeline`)
- Tree models: Random Forest, LightGBM, XGBoost
- Metric: **ROC-AUC** and **PR-AUC**. Accuracy is not used because of class imbalance.

## Results

_Fill in after modelling._

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression | … | … |
| Random Forest | … | … |
| LightGBM | … | … |
| XGBoost | … | … |

## Project structure

```
customer-churn/
├── data_processing/      # load_data.py, preprocess.py, feature_engineering.py
├── model/                # train.py, evaluate.py, predict.py
├── pipeline/             # data_pipeline.py, training_pipeline.py
├── components/           # data_ingestion.py, model_builder.py, evaluation.py
├── notebooks/            # exploratory_analysis.ipynb, model_training.ipynb
├── sql/                  # SQL feature queries
├── utils/                # helper_functions.py, config.py, logging_setup.py, exception_handling.py
├── reports/figures/      # saved charts
├── data/raw/             # raw data (not committed)
├── app.py
├── requirements.txt
└── README.md
```

## How to run

```bash
git clone https://github.com/abdallahh07/customer-churn.git
cd customer-churn

conda create -n customer-churn python=3.11 -y
conda activate customer-churn
pip install -r requirements.txt
```

1. Download `online_retail_II.csv` from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (also on Kaggle as "Online Retail II UCI"), and place it in `data/raw/`.
2. Run `notebooks/exploratory_analysis.ipynb`, then `notebooks/model_training.ipynb`.

## Limitations and next steps

_Fill in at the end, e.g. data covers only two years, no marketing or customer-service data, single retailer._

---

## Author

**Abdallah Hashad**: junior data scientist with a background in law and finance.
[LinkedIn](https://www.linkedin.com/in/abdallah-hashad-1a6254249) · [GitHub](https://github.com/abdallahh07)
