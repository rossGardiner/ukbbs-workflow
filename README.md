# UK BBS workflow

A reproducible UK BBS preprocessing notebook plus a **small, model-agnostic
forecasting interface**. Keep one notebook per experiment/model; share the
split, fit/predict, and evaluation contract in `src/ukbbs/`.

## Setup

Python >= 3.11, from the project root:

```bash
python -m pip install -e '.[notebook]'
python -m jupyter lab
```

Run `notebooks/01_download_and_preprocess.ipynb` first to generate
`data/processed/ukbbs_model_data.csv` (raw files are downloaded by that
notebook). Then open `notebooks/02_baseline_forecasting.ipynb` for a complete
baseline experiment. Processed/raw datasets are intentionally not in Git.

Run the offline tests with:

```bash
python -m unittest discover -s tests -v
```

## Standard forecasting contract

```python
from ukbbs import (
    ForecastTask, LastValueForecaster, chronological_split, run_forecast,
)

task = ForecastTask()  # site + species, year, original UKBBS count column
train, test = chronological_split(observations, train_through=2018, task=task)
report = run_forecast(LastValueForecaster(), train, test, task)
print(report.metrics)         # n_scored, n_unscored, mae, rmse, bias
print(report.predictions)     # group/time keys, observed, prediction, residual
```

Or, to work directly with the same components in any notebook:

```python
from ukbbs import evaluate_forecasts

model = LastValueForecaster().fit(train, task)
future = test.drop(columns=[task.target])  # never pass the answer to the model
predictions = model.predict(future)
report = evaluate_forecasts(test, predictions, task)
```

**To add another method**, implement two methods on any class:

- `fit(train: pd.DataFrame, task: ForecastTask) -> self`: train the method.
- `predict(future: pd.DataFrame) -> pd.DataFrame`: return exactly one row per
  future key (`task.keys`) plus a finite numeric `prediction` column. The
  `future` frame has no target and may have extra covariates.

This is a structural `ForecastModel` protocol: you don't need to subclass it.
You can wrap ARIMA, scikit-learn, PyTorch, or a custom method in a class with
these methods. Models are responsible for any required feature engineering.
`run_forecast` works with any such model. Use `ForecastTask(target=...,
 time_col=..., group_cols=(...))` for regional or alternative targets.

`evaluate_forecasts` joins on keys, **not** row order, and rejects duplicate,
missing, or unmatched keys and invalid predictions. Unobserved (null) test
labels are reported but not scored; an observed count of zero **is** scored.
Pass `metrics={"my_metric": callable}` to replace the built-in metrics;
callables receive `(actual, prediction)` numeric NumPy arrays.

## UK BBS and forecasting caveats

- The preprocessor emits *observed* species–square–year rows. It does not
  manufacture counts of zero for missing records. Use the notebook's
  `sampled_square_years.csv` when separately deciding whether non-detections
  can be represented as zeros. Never treat an unsurveyed year as zero.
- A group's last observed target is carried forward by the example baseline.
  Groups with no training history raise an error. In the baseline notebook,
  we explicitly restrict evaluation to previously seen site/species groups.
  Other methods can implement cold-start prediction if desired.
- The chronological split uses actual calendar years and does not silently
  compress omitted years (including excluded years 2001 and 2020).
- For metadata experiments, add feature columns to train and future rows, but
  use only features that **would be available at the forecast origin**. The
  framework hides the test target; it cannot automatically detect leakage
  through future-unknown covariates.
- A `run_forecast` call requires every training timestamp to precede every
  test timestamp. This covers standard holdout experiments; for rolling-origin
  backtesting, call it separately for each chronological fold and aggregate
  the resulting evaluations in your notebook.

This patch only introduces the forecasting layer and example notebook. It does
not alter the original downloaded source, preprocessing decisions, or the
existing notebook's data-release metadata.
