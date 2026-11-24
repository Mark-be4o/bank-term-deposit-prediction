# Bank Term Deposit Prediction

Bank marketing analysis and term deposit subscription prediction, plus a small app script.

## What it does

Full analysis of the bank marketing data (4,521 rows, 17 columns): cleaning, EDA, and models that predict whether a client subscribes to a term deposit. app.py is a small app wrapper around the model.

## Files

- `bank_term_deposit_prediction.ipynb`
- `app.py`

## Tools

pandas, numpy, scikit-learn, TensorFlow, Plotly, seaborn, matplotlib

## Dataset

UCI Bank Marketing dataset (bank.csv, semicolon separated), not included. Source: https://archive.ics.uci.edu/dataset/222/bank+marketing

The dataset is not included in this repo.

## Results

The saved notebook copy has no recorded metrics. Run it to reproduce them.

## How to run

1. Open the notebook in Google Colab or Jupyter.
2. Install dependencies: `pip install -r requirements.txt`.
3. Download the dataset from the link above (some notebooks fetch it with `kagglehub`, which asks for Kaggle credentials) and update the data path if needed.
4. Run all cells.

## Author

Mark Ayman
