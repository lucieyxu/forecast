from typing import Dict, List

from kfp.dsl import component


@component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-bigquery",
        "google-cloud-aiplatform",
        "google-cloud-storage",
    ],
)
def predictions(
    project_id: str,
    bq_dataset_name: str,
    experiment_name: str,
    experiment_run_name: str,
    model_type: str,
    time_column: str,
    series_column: str,
    target_column: str,
    attributes_columns: List[str],
    covariates_columns_known: List[str],
    covariates_columns_unknown: List[str],
    context_window: int,
    forecast_granularity: str,
    fold: int,
    pipeline_bucket: str,
    rolling_window_time_split_gcs_name: str,
) -> Dict[str, float]:
    """Create evaluation Bigquery table with metrics and optionally generate artifacts

    Args:
        project_id (str): project ID
        bq_dataset_name (str): BigQuery dataset name
        experiment_name (str): VertexAI experiment name
        experiment_run_name (str): VertexAI experiment run name
        model_type (str): Model type
        time_column (str): Timestamp column to use for forecast
        series_column (str): Column to use to get the forecast granularity. Each unique ID will be a time series
        target_column (str): Column with value to forecast
        attributes_columns (List[str]): Attributes of time series to use in training (static)
        covariates_columns_known (List[str]): Covariates known at time of prediction (Promotions for example)
        covariates_columns_unknown (List[str]): Covariates unknown at time of prediction (stock for example)
        context_window (int): Training context window
        forecast_granularity (str): DAILY or WEEKLY
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
    start_test_date = time_split["test_start_date"]

    bq_client = bigquery.Client(project=project_id)

    if forecast_granularity == "WEEK":
        time_divisor = 7
    elif forecast_granularity == "DAY":
        time_divisor = 1
    else:
        raise ValueError(f"Granularity not supported {forecast_granularity}")

    display_name: str
    if model_type == "AutoMLForecastingTrainingJob":
        display_name = (
            f"l2l_context{context_window}_{experiment_name}_{experiment_run_name}"
        )
    elif model_type == "TimeSeriesDenseEncoderForecastingTrainingJob":
        display_name = (
            f"tide_context{context_window}_{experiment_name}_{experiment_run_name}"
        )
    else:
        raise ValueError(f"Model not supported {model_type}")

    covariates_sql = ", ".join(
        attributes_columns
        + covariates_columns_known
        + covariates_columns_unknown
    )

    destination_table_id = (
        f"{project_id}.{bq_dataset_name}.{experiment_name}_eval_looker"
    )

    query = f"""
    WITH eval_casted as (
        SELECT *,
            CAST({target_column} as FLOAT64) AS actual_{target_column},
        FROM `{project_id}.{bq_dataset_name}.{experiment_name}_{experiment_run_name}_eval`
    )
    SELECT
        DATE({time_column}) as {time_column},
        DATE(predicted_on_{time_column}) as predicted_on_{time_column},
        actual_{target_column},
        {series_column},
        predicted_{target_column}.value as predicted_{target_column},
        {covariates_sql},
        '{display_name}' as model_name,
        '{experiment_run_name.split("-fold-")[0]}' AS run_id,
        {fold} as fold,
        CURRENT_TIMESTAMP() AS eval_timestamp,
        ROUND(EXTRACT(DAY from DATE({time_column}) - "{start_test_date}") / {time_divisor}) as lag,
        (actual_{target_column} - predicted_{target_column}.value) as diff,
        ABS(actual_{target_column} - predicted_{target_column}.value) * actual_{target_column} as wAbsError,
        actual_{target_column} * actual_{target_column} as wHist,
    FROM eval_casted
    WHERE {time_column} = predicted_on_{time_column}
    ORDER BY {series_column}, {time_column}
    """
    logger.info(query)
    # Append to looker table if already created, we don't want to remove data
    # From other model trainings
    job_config = bigquery.QueryJobConfig(
        destination=destination_table_id,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
    )

    job = bq_client.query(query=query, job_config=job_config)
    job.result()
    logger.info(
        f"Eval data created in {(job.ended-job.started).total_seconds()} s"
    )  # type: ignore
