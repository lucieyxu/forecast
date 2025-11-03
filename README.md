# Forecast
Code templates for demand forecasting projects

## Context
we take the example of Citbike forecasting using the public dataset in `bigquery-public-data.new_york.citibike_trips`.

## Run pipelines

### ARIMA pipeline
```
export PROJECT_ID=[REPLACE_WITH_PROJECT_ID]
export PIPELINE_BUCKET=[REPLACE_WITH_PIPELINE_BUCKET]
poetry install
poetry run python pipelines/arima_pipeline.py --experiment pipelines/experiments/arima_experiment.yaml 
```