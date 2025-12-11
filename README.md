# Context
## Objectives
By following the associated Google doc [Forecast Project Kickstart Framework](https://docs.google.com/document/d/1SkI52LpgpZnXg1prMkypPnv5R3uHlAENzuSgKPmiXaY/edit?pli=1&resourcekey=0-kuJx9W1FsnYoCbCz73jzhA&tab=t.0#heading=h.8xyn7r6m6bck) and this code repo, you will achieve the following:
* Standardized Workflow: Move from Exploratory Data Analysis (EDA) to Model Evaluation using a consistent, repeatable framework.
* Robust Evaluation: Implement a Rolling Window backtesting strategy to accurately simulate production performance, accounting for seasonality and trend shifts.
* Model Diversity: Leverage a "champion/challenger" approach by easily comparing statistical models (BigQuery ARIMA+) against Deep Learning architectures (Vertex AI TIDE, AutoML L2L).
* Explainability: Ensure model transparency by generating feature attribution metrics that Business stakeholders can trust and using Looker studio dashboards to communicate results.

The CitiBike example is used to ensure concepts are concrete and reproducible: Forecasting weekly New York CitiBike rentals.
We utilize the public dataset `bigquery-public-data.new_york.citibike_trips` to demonstrate every step of the pipeline, from raw data ingestion to dashboard visualization in Looker Studio.


# Run pipelines

## Poetry setup
```
export PROJECT_ID=[REPLACE_WITH_PROJECT_ID]
export PIPELINE_BUCKET=[REPLACE_WITH_PIPELINE_BUCKET]
poetry install
```
## ARIMA+ model training and evaluation
```
poetry run python pipelines/arima_pipeline.py --experiment pipelines/experiments/arima_experiment.yaml 
```

## ARIMA+ XREG model training and evaluation
```
poetry run python pipelines/arima_pipeline.py --experiment pipelines/experiments/arima_xreg_experiment.yaml
```

## AutoML L2L model training and evaluation
```
poetry run python pipelines/arima_pipeline.py --experiment pipelines/experiments/l2l_experiment.yaml
```

## TiDE model training and evaluation
```
poetry run python pipelines/arima_pipeline.py --experiment pipelines/experiments/tide_experiment.yaml
```