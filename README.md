# 🕒 Time Series Analysis and Demand Forecasting

## 📌 Project Overview

This project presents a comprehensive **time series analysis and demand forecasting** pipeline using historical Walmart sales data. The goal is to evaluate and compare multiple forecasting techniques, enabling accurate sales prediction and insightful business decisions.

---

## 📊 Data Sources

- **train.csv**: Weekly sales by store and department
- **stores.csv**: Metadata including store type and size
- **features.csv**: External features such as temperature, fuel price, CPI, unemployment, and markdowns

[Dataset from Kaggle](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast/data)

---

## 🧹 Data Preparation

- Merging datasets on `Store` and `Date`
- Handling missing and outlier values
- Numeric encoding of categorical variables
- Feature optimization for CPI, Unemployment, and MarkDowns

---

## 📈 Exploratory Data Analysis (EDA)

- Seasonal trends around holidays (Thanksgiving, Black Friday, Christmas)
- Quarterly/Monthly/yearly sales breakdown
- Correlation between markdowns and store size
- Minimal influence from external economic indicators

---

## 🧰 Feature Engineering

- **Temporal features**: day, week, month, quarter, year
- **Rolling stats**: moving average and std. deviation
- **Lag features**: up to 52 weeks
- **Holiday proximity**: days to/from holidays
- **Sine/cosine seasonal encodings**

---

## 🤖 Modeling Approaches

### ✅ XGBoost
- Highest accuracy (~98%)
- Full use of engineered features
- Best performer in RMSE, MAE, R²

### 📉 SARIMA
- Univariate model on aggregated sales
- Reasonable performance but outperformed by ML models

### 🧠 LSTM Neural Network
- Complex architecture with seasonal/holiday features
- Good accuracy and robust forecasts with uncertainty intervals

### 📆 Facebook Prophet
- Intuitive and interpretable
- Handles multiple seasonalities and holidays
- Good performance, though less accurate than XGBoost/LSTM

---

## 📊 Model Comparison

| Model    | RMSE ↓ | MAE ↓ | R² ↑ |
|----------|--------|-------|------|
| XGBoost | ✅ Best | Best  | Best |
| LSTM    | High   | High  | High |
| Prophet | Medium | Medium| Medium |
| SARIMA  | Low    | Low   | Low  |

---

## 📝 Conclusions

- **XGBoost** is the best model for this data
- Feature engineering plays a critical role in performance
- Sales are highly seasonal and promotion-driven
- Future improvements could include real-time pipelines and external economic APIs

---


