# Demand Forecasting with Time Series and Machine Learning

This project explores weekly Walmart sales forecasting with a notebook-driven workflow that compares classical time series methods and machine learning models. The focus is on feature engineering, seasonality, holiday effects, and model comparison for retail demand planning.

## What this project includes

- Historical Walmart sales data
- Feature engineering for weekly forecasting
- Exploratory analysis of seasonality and promotions
- Comparison across XGBoost, SARIMA, Prophet, and LSTM-based approaches
- A notebook that consolidates the full analysis flow

## Dataset

Files under `dataset/`:

- `train.csv`
- `test.csv`
- `stores.csv`
- `features.csv`

Source: [Walmart Sales Forecast dataset on Kaggle](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast/data)

## Repository structure

```text
Demand-Forecasting-Project/
├── Demand Forecasting.ipynb
├── dataset/
├── requirements.txt
├── requirements-deep-learning.txt
├── requirements-prophet.txt
└── README.md
```

## Main modeling ideas

- Temporal feature extraction from weekly sales dates
- Lag and rolling-window features
- Holiday-aware demand patterns
- Comparison between statistical, boosting, and neural approaches

## Reported outcome

- `XGBoost` is the strongest overall performer in the current notebook
- `SARIMA` provides a simpler baseline
- `Prophet` adds interpretability and multi-seasonality support
- `LSTM` captures longer temporal structure but requires a heavier environment

## Local setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install the base dependencies used by most of the notebook:

```bash
pip install -r requirements.txt
```

3. Install optional model stacks only if you plan to run those sections:

```bash
pip install -r requirements-prophet.txt
pip install -r requirements-deep-learning.txt
```

4. Launch Jupyter:

```bash
jupyter notebook
```

5. Open `Demand Forecasting.ipynb`.

## Dependency notes

- `Prophet` can require additional local build tooling depending on your Python version
- `TensorFlow` is optional here because it is only needed for the LSTM section
- Using separate requirement files keeps the notebook easier to start on fresh machines

## Portfolio talking points

- Strong retail forecasting use case with real business relevance
- Combines feature engineering with multiple forecasting paradigms
- Shows practical tradeoffs between interpretability, complexity, and accuracy
- Useful example of notebook-based experimentation for demand planning

## Recommended next improvements

- Export final evaluation tables and charts as versioned assets
- Add a lightweight data dictionary for the Walmart fields
- Break the notebook into reusable pipeline scripts
- Save trained model artifacts for reproducible forecasting demos
