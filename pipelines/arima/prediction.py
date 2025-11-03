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
def predictions(
    project_id: str,
    bq_dataset_name: str,
    bq_table_prepped: str,
    bq_model_name: str,
    experiment_name: str,
    experiment_run_name: str,
    model_type: str,
    time_column: str,
    series_column: str,
    target_column: str,
    covariates: List[str],
    forecast_granularity: str,
    options: Dict[str, Any],
    fold: int,
    pipeline_bucket: str,
    rolling_window_time_split_gcs_name: str,
) -> None:
    """Merge Historical data and forecast predictions and export to Bigquery.

    Args:
        project_id (str): project ID
        bq_dataset_name (str): BigQuery dataset name
        bq_table_prepped (str): BigQuery table to create or replace
        bq_model_name (str): BigQuery model name
        experiment_name (str): VertexAI experiment name
        experiment_run_name (str): VertexAI experiment run name
        model_type (str): Model type
        time_column (str): Timestamp column to use for forecast
        series_column (str): Column to use to get the forecast {series_column}. Each unique ID will be a time series
        target_column (str): Column with value to forecast
        forecast_granularity (str): DAILY or WEEKLY
        options (Dict[str, Any]): Model options
        fold (int): when using rolling window training pipeline
        pipeline_bucket (str): pipeline bucket
        rolling_window_time_split_gcs_name (str): time splits gcs name
    """
    import json
    import logging

    from google.cloud import bigquery, storage

    logger = logging.getLogger(__name__)

    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(pipeline_bucket)
    blob = bucket.blob(rolling_window_time_split_gcs_name)
    time_split = json.loads(blob.download_as_string())[fold]
    forecast_test_length = time_split["test_length"]
    start_test_date = time_split["test_start_date"]
    end_test_date = time_split["test_end_date"]

    bq_client = bigquery.Client(project=project_id)
    destination_table_id = (
        f"{project_id}.{bq_dataset_name}.{experiment_name}_eval_looker"
    )

    time_divisor: int
    if forecast_granularity == "WEEKLY":
        time_divisor = 7
    elif forecast_granularity == "DAILY":
        time_divisor = 1
    else:
        raise ValueError(f"Granularity not supported {forecast_granularity}")

    if model_type == "ARIMA_PLUS":
        forecast_args_sql = (
            f"STRUCT({forecast_test_length} AS horizon, 0.9 AS confidence_level)"
        )
    else:
        variables = f'{series_column}, {time_column}, {", ".join(covariates)}'
        forecast_args_sql = f"""STRUCT({forecast_test_length} AS horizon, 0.9 AS confidence_level),
            (
                SELECT
                    {variables}
                FROM
                    `{project_id}.{bq_dataset_name}.{bq_table_prepped}`
            )"""

    if (
        "FORECAST_LIMIT_LOWER_BOUND" in options
        or "FORECAST_LIMIT_UPPER_BOUND" in options
    ):
        forecast_sql = f"""SELECT
                *,
                EXTRACT(DATE from forecast_timestamp) as {time_column}
            FROM ML.FORECAST(
                MODEL `{project_id}.{bq_dataset_name}.{bq_model_name}`,
                {forecast_args_sql})
            WHERE time_series_type = 'forecast'
            AND EXTRACT(DATE from time_series_timestamp) >= "{start_test_date}" 
            AND EXTRACT(DATE from time_series_timestamp) < "{end_test_date}"
            ORDER BY time_series_timestamp
            """
    else:
        forecast_sql = f"""SELECT
                *,
                EXTRACT(DATE from time_series_timestamp) as {time_column},
                time_series_adjusted_data as forecast_value
            FROM ML.EXPLAIN_FORECAST(
                MODEL `{project_id}.{bq_dataset_name}.{bq_model_name}`,
                {forecast_args_sql})
            WHERE time_series_type = 'forecast'
            AND EXTRACT(DATE from time_series_timestamp) >= "{start_test_date}" 
            AND EXTRACT(DATE from time_series_timestamp) < "{end_test_date}"
            ORDER BY time_series_timestamp
        """

    query = f"""
        -- Create scaffold of weekly dates for test set
        -- We need to have 0 as qty for weeks that are missing for a given time series
        WITH GranularityAttributes AS (
            SELECT
            {series_column},
            FROM
            `{project_id}.{bq_dataset_name}.{bq_table_prepped}`
            GROUP BY
            {series_column}
        ),

        DateRangeScaffold AS (
        SELECT
            g.*,
            TIMESTAMP(w.week_start) AS {time_column}
        FROM
            GranularityAttributes AS g
            CROSS JOIN
            (SELECT week_start 
            FROM UNNEST(
                GENERATE_DATE_ARRAY('{start_test_date}', '{end_test_date}',
                    INTERVAL 1 WEEK)) AS week_start
            ) AS w
        ),

        -- Get data within test set
        DataInRange AS (
        SELECT *
        FROM `{project_id}.{bq_dataset_name}.{bq_table_prepped}`
        WHERE {time_column} >= '{start_test_date}'
        AND {time_column} < '{end_test_date}'
        ),

        hist as (
            -- Join the weekly scaffold with the existing data to fill in the missing rows.
            SELECT
            scaffold.*,
            COALESCE(data.{target_column}, 0) AS {target_column},
            FROM DateRangeScaffold AS scaffold
            LEFT JOIN DataInRange AS data
            ON scaffold.{series_column} = data.{series_column}
            AND scaffold.{time_column} = data.{time_column}
        ),

        FORECAST AS (
            {forecast_sql}
        )

        SELECT t2.{time_column} as {time_column}, t2.* EXCEPT ({target_column}, {time_column}), {target_column} as actualQty, 
        CASE
            WHEN forecast_value IS NULL THEN 0
            WHEN forecast_value < 0 THEN 0
            ELSE forecast_value
        END AS forecastQty,
        CASE
            WHEN prediction_interval_lower_bound IS NULL THEN 0
            WHEN prediction_interval_lower_bound < 0 THEN 0
            ELSE prediction_interval_lower_bound
        END AS predicted_p10,
        CASE
            WHEN prediction_interval_upper_bound IS NULL THEN 0
            WHEN prediction_interval_upper_bound < 0 THEN 0
            ELSE prediction_interval_upper_bound
        END AS predicted_p90,
        '{bq_model_name}' as model_name,
        '{experiment_run_name.split("-fold-")[0]}' AS run_id,
        {fold} as fold,
        CURRENT_TIMESTAMP() AS eval_timestamp,
        ROUND(EXTRACT(DAY from DATE(t2.{time_column}) - "{start_test_date}") / {time_divisor}) as lag,
        CASE
            WHEN forecast_value IS NULL THEN ({target_column})
            WHEN forecast_value < 0 THEN ({target_column})
            ELSE ({target_column} - forecast_value)
        END AS diff,
        CASE
            WHEN forecast_value IS NULL THEN ABS({target_column}) * {target_column}
            WHEN forecast_value < 0 THEN ABS({target_column}) * {target_column}
            ELSE ABS({target_column} - forecast_value) * {target_column}
        END AS wAbsError,
        {target_column} * {target_column} as wHist,
        FROM FORECAST as t1
        RIGHT JOIN hist as t2 
            ON t1.{series_column} = t2.{series_column}
            AND t1.{time_column} = DATE(t2.{time_column})
        ORDER BY t2.{time_column}
    """
    logger.info(query)
    # Append to looker table if already created, we don't want to remove data
    # From other model trainings
    job_config = bigquery.QueryJobConfig(
        destination=destination_table_id,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )

    job = bq_client.query(query=query, job_config=job_config)
    job.result()
    logger.info(
        f"Eval data created in {(job.ended-job.started).total_seconds()} s"
    )  # type: ignore
