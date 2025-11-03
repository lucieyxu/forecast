from typing import Any, Dict, List

from kfp.dsl import component


@component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-bigquery",
        "google-cloud-aiplatform",
        "google-cloud-storage",
    ],
)
def train_arima_model(
    project_id: str,
    bq_dataset_name: str,
    bq_table_prepped: str,
    bq_model_name: str,
    experiment_name: str,
    experiment_description: str,
    experiment_run_name: str,
    model_type: str,
    target_column: str,
    time_column: str,
    series_column: str,
    split_column: str,
    forecast_granularity: str,
    options: Dict[str, Any],
    covariates: List[str],
    fold: int,
    pipeline_bucket: str,
    rolling_window_time_split_gcs_name: str,
) -> None:
    """ARIMA+ or ARIMA+ xreg BQML implementation

    Args:
        project_id (str): project ID
        bq_dataset_name (str): BigQuery dataset name
        bq_table_prepped (str): BigQuery table to create or replace
        bq_model_name (str): BigQuery model name
        experiment_name (str): VertexAI experiment name
        experiment_description (str): VertexAI experiment description
        experiment_run_name (str): VertexAI experiment run name
        model_type (str): Model type
        target_column (str): Column with value to forecast
        time_column (str): Timestamp column to use for forecast
        series_column (str): Column to use to get the forecast granularity. Each unique ID will be a time series
        split_column (str): Column containing categories for train val test
        forecast_granularity (str): DAILY or WEEKLY
        options (Dict[str, Any]): Model options
        covariates (List[str]): Model covariates to use
        fold (int): when using rolling window training pipeline
        pipeline_bucket (str): pipeline bucket
        rolling_window_time_split_gcs_name (str): time splits gcs name
    """
    import json
    import logging

    from google.cloud import aiplatform, bigquery, storage

    logger = logging.getLogger(__name__)

    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(pipeline_bucket)
    blob = bucket.blob(rolling_window_time_split_gcs_name)
    time_split = json.loads(blob.download_as_string())[fold]
    forecast_horizon_length = time_split['test_length']


    options_sql = []
    for k, v in options.items():
        options_sql.append(f"{k} = {v}")
    options_sql_str = ", ".join(options_sql)

    if model_type == "ARIMA_PLUS" or len(covariates) == 0:
        variables = f"{series_column}, {time_column}, {target_column}"
    else:
        variables = f'{series_column}, {time_column}, {target_column}, {", ".join(covariates)}'

    # BQML training
    bq_client = bigquery.Client(project=project_id)
    query = f"""
        CREATE OR REPLACE MODEL `{project_id}.{bq_dataset_name}.{bq_model_name}`
        OPTIONS
        (
            model_type = '{model_type}',
            time_series_timestamp_col = '{time_column}',
            time_series_data_col = '{target_column}',
            time_series_id_col = '{series_column}',
            data_frequency = '{forecast_granularity}',
            horizon = {forecast_horizon_length},
            {options_sql_str}
        ) AS
        SELECT {variables}
        FROM `{project_id}.{bq_dataset_name}.{bq_table_prepped}`
        WHERE {split_column} in ('TRAIN','VALIDATE')
    """
    logger.info(query)
    job = bq_client.query(query)
    job.result()
    train_time = (job.ended - job.started).total_seconds()  # type: ignore
    logger.info(f"Model training query completed in {train_time} seconds.")

    # Export parameters to Vertex AI Experient
    aiplatform.init(
        experiment=experiment_name,
        experiment_description=experiment_description,
    )

    params_to_log = {k: str(v) for k, v in options.items()}
    params_to_log.update(
        {
            "model_type": model_type,
            "horizon": str(forecast_horizon_length),
            "data_frequency": forecast_granularity,
            "model_id": f"{project_id}.{bq_dataset_name}.{bq_model_name}",
        }
    )
    logger.info(params_to_log)
    with aiplatform.start_run(experiment_run_name) as run:
        run.log_params(params_to_log)
        run.log_metrics({"train_time": train_time})
