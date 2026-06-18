# Housing price regression

Regression pipeline on the California Housing dataset using scikit-learn.
Trains three models — ridge regression, random forest, and gradient boosting — then reports RMSE, MAE, and R² with cross-validation and saves diagnostic plots.

## Structure

```
train.py          # end-to-end training entrypoint
src/
  data.py         # dataset loading, train/test split, scaling
  features.py     # derived feature engineering
  models.py       # model training wrappers
  evaluate.py     # metrics and cross-validation
  plot.py         # feature importance and residual plots
notebooks/        # EDA notebooks (added later)
outputs/          # saved figures from train.py
data/             # local data cache (gitignored)
```

## Dataset

California Housing from `sklearn.datasets.fetch_california_housing`. No external download needed.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python train.py
```

Prints a metrics table and writes plots to `outputs/`.
