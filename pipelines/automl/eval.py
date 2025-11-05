from typing import Dict

from kfp.dsl import component


@component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-bigquery",
        "google-cloud-aiplatform",
        "google-cloud-storage",
        "pandas",
        "db-dtypes",
        "tabulate",
    ],
)
def evaluate_model(
    project_id: str,
    region: str,
    bq_dataset_name: str,
    experiment_name: str,
    experiment_description: str,
    experiment_run_name: str,
    time_column: str,
    series_column: str,
    target_column: str,
) -> Dict[str, float]:
    """Create evaluation metrics and log to Vertex AI experiments

    Args:
        project_id (str): project ID
        region (str): VAI Experiment region
        bq_dataset_name (str): BigQuery dataset name
        experiment_name (str): VertexAI experiment name
        experiment_description (str): VertexAI experiment description
        experiment_run_name (str): VertexAI experiment run name
        time_column (str): Timestamp column to use for forecast
        series_column (str): Column to use to get the forecast granularity. Each unique ID will be a time series
        target_column (str): Column with value to forecast
    """
    import json
    import logging

    from google.cloud import aiplatform, bigquery, storage

    logger = logging.getLogger(__name__)
    bq_client = bigquery.Client(project=project_id)

    query = f"""
        WITH
            FORECASTS AS (
                SELECT
                    DATE({time_column}) as {time_column},
                    DATE(predicted_on_{time_column}) as predicted_on_{time_column},
                    CAST({target_column} as INT64) AS {target_column},
                    {series_column},
                    predicted_{target_column}.value as predicted_{target_column}
                FROM `{project_id}.{bq_dataset_name}.{experiment_name}_{experiment_run_name}_eval`
                WHERE {time_column} = predicted_on_{time_column}
            ),
            FILL_NULL AS (
                SELECT *,
                CASE
                    WHEN predicted_{target_column} IS NULL THEN 0
                    WHEN predicted_{target_column} < 0 THEN 0
                    ELSE predicted_{target_column}
                END AS forecast_value_filled
                FROM FORECASTS
            ),
            DIFFS AS (
                SELECT 
                    {series_column},
                    {time_column},
                    'forecast' as time_series_type,
                    forecast_value_filled as forecast_value,
                    {target_column} as actual_value,
                    ({target_column} - predicted_{target_column}) as diff,
                    ABS({target_column} - predicted_{target_column}) * {target_column} as wAbsError,
                    {target_column} * {target_column} as wHist,
                FROM FILL_NULL    
            )
        SELECT
            time_series_type, 
            SUM(wAbsError) * 100.0 / SUM(wHist) as wMAPE,
            AVG(SAFE_DIVIDE(ABS(diff), actual_value)) as MAPE,
            AVG(ABS(diff)) as MAE,
            SAFE_DIVIDE(SUM(ABS(diff)), SUM(actual_value)) as pMAE,
            AVG(POW(diff, 2)) as MSE,
            SQRT(AVG(POW(diff, 2))) as RMSE,
            SAFE_DIVIDE(SQRT(AVG(POW(diff, 2))), AVG(actual_value)) as pRMSE
        FROM DIFFS
        GROUP BY
            time_series_type
        ORDER BY
            time_series_type  
    """
    logger.info(query)
    customMetricsOverall = bq_client.query(query=query).to_dataframe()
    logger.info(f"Overall metrics: {customMetricsOverall.to_markdown()}")  # type: ignore

    # Log metrics to Vertex AI experiments
    aiplatform.init(
        project=project_id,
        location=region,
        experiment=experiment_name,
        experiment_description=experiment_description,
    )
    metrics = customMetricsOverall.iloc[0].to_dict()
    logger.info(f"Overall metrics: {metrics}")
    del metrics["time_series_type"]
    with aiplatform.start_run(experiment_run_name, resume=True) as run:
        run.log_metrics(metrics)
    return metrics
