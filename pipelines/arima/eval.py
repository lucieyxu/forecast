from typing import Any, Dict, List

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
    location: str,
    bq_dataset_name: str,
    bq_table_prepped: str,
    bq_model_name: str,
    experiment_name: str,
    experiment_description: str,
    experiment_run_name: str,
    model_type: str,
    time_column: str,
    series_column: str,
    target_column: str,
    split_column: str,
    covariates: List[str],
    forecast_granularity: str,
    options: Dict[str, Any],
    fold: int,
    pipeline_bucket: str,
    rolling_window_time_split_gcs_name: str,
) -> None:
    """Create evaluation Bigquery table with metrics

    Args:
        project_id (str): project ID
        location (str): Location
        bq_dataset_name (str): BigQuery dataset name
        bq_table_prepped (str): BigQuery table to create or replace
        bq_model_name (str): BigQuery model name
        experiment_name (str): VertexAI experiment name
        experiment_description (str): VertexAI experiment description
        experiment_run_name (str): VertexAI experiment run name
        model_type (str): Model type
        time_column (str): Timestamp column to use for forecast
        series_column (str): Column to use to get the forecast granularity. Each unique ID will be a time series
        target_column (str): Column with value to forecast
        split_column (str): Column containing categories for train val test
        forecast_granularity (str): DAILY or WEEKLY
        options (Dict[str, Any]): Model options
        fold (int): when using rolling window training pipeline
        pipeline_bucket (str): pipeline bucket
        rolling_window_time_split_gcs_name (str): time splits gcs name
    """
    import json
    import logging

    import numpy as np
    import pandas as pd
    from google.cloud import aiplatform, bigquery, storage

    logger = logging.getLogger(__name__)
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(pipeline_bucket)
    blob = bucket.blob(rolling_window_time_split_gcs_name)
    time_split = json.loads(blob.download_as_string())[fold]
    forecast_test_length = time_split["test_length"]
    start_test_date = time_split["test_start_date"]
    end_test_date = time_split["test_end_date"]

    bq_client = bigquery.Client(project=project_id)
    bq_eval_table_id = f"{experiment_run_name}_eval"

    if forecast_granularity == "WEEKLY":
        time_divisor = 7
    elif forecast_granularity == "DAILY":
        time_divisor = 1
    else:
        raise ValueError(f"Granularity not supported {forecast_granularity}")

    if model_type == "ARIMA_PLUS":
        forecast_args_sql = (
            f"STRUCT({forecast_test_length} AS horizon, 0.95 AS confidence_level)"
        )
    else:
        variables = f'{series_column}, {time_column}, {", ".join(covariates)}'
        forecast_args_sql = f"""STRUCT({forecast_test_length} AS horizon, 0.95 AS confidence_level),
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
                EXTRACT(DATE from forecast_timestamp) as f_{time_column}
            FROM ML.FORECAST(
                MODEL `{project_id}.{bq_dataset_name}.{bq_model_name}`,
                {forecast_args_sql})"""
    else:
        forecast_sql = f"""SELECT
                *,
                EXTRACT(DATE from time_series_timestamp) as f_{time_column},
                time_series_adjusted_data as forecast_value
            FROM ML.EXPLAIN_FORECAST(
                MODEL `{project_id}.{bq_dataset_name}.{bq_model_name}`,
                {forecast_args_sql})"""

    # Forecast values can be missing for the test set for a granularity if the
    # time series did not have any data points in the training set (cold start).
    # We also use a left join to join actual and forecast, since if we don't have
    # any actual value for the week, wHist = wAbsError = 0 and we cannot divide
    # by 0 for wMAPE.
    # TODO: change to full join if we want to compute other metrics than wMAPE.
    query = f"""
        CREATE OR REPLACE TABLE `{project_id}.{bq_dataset_name}.{bq_eval_table_id}` AS
        WITH FORECAST AS (
            {forecast_sql}
        ),
        ACTUAL AS (
            SELECT {series_column} as a_{series_column}, EXTRACT(DATE from {time_column}) as {time_column}, sum({target_column}) as actual_value
            FROM `{project_id}.{bq_dataset_name}.{bq_table_prepped}`
            GROUP BY {series_column}, {time_column}
        ),
        COMBINED AS (
            SELECT *
            FROM ACTUAL t1
            LEFT JOIN FORECAST t2
            ON t1.{time_column} = t2.f_{time_column}
            AND t1.a_{series_column} = t2.{series_column}
        ),
        FILL_NULL AS (
            SELECT *,
            CASE
                WHEN forecast_value IS NULL THEN 0
                WHEN forecast_value < 0 THEN 0
                ELSE forecast_value
            END AS forecast_value_filled
            FROM COMBINED
        ),
        DIFFS AS (
            SELECT *,
            (actual_value - forecast_value_filled) as diff,
            ABS(actual_value - forecast_value_filled) * actual_value as wAbsError,
            actual_value * actual_value as wHist,
            FROM FILL_NULL
        )
        SELECT *, wAbsError * 100.0 / wHist as wMAPE,
        ROUND(EXTRACT(DAY from {time_column} - "{start_test_date}") / {time_divisor}) as lag
        FROM DIFFS
        ORDER BY {time_column}
    """
    logger.info(query)
    job = bq_client.query(query)
    job.result()
    logger.info(
        f"Evaluation run in {(job.ended-job.started).total_seconds()} s"  # type: ignore
    )

    query = f"""
        SELECT SUM(wAbsError) * 100.0 / SUM(wHist) as wMAPE
        FROM `{project_id}.{bq_dataset_name}.{bq_eval_table_id}`
        WHERE {time_column} >= "{start_test_date}" and {time_column} < "{end_test_date}"
        AND time_series_type = "forecast"
    """
    logger.info(query)
    job = bq_client.query(query)
    metrics_df = job.to_dataframe()
    logger.info("Evaluation metrics:")
    logger.info(
        metrics_df.to_markdown(
            index=False, numalign="left", stralign="left"
        )
    )
    aiplatform.init(
        experiment=experiment_name,
        experiment_description=experiment_description,
    )
    with aiplatform.start_run(experiment_run_name, resume=True) as run:
        # Log metrics
        run.log_metrics({"Global_wMAPE": metrics_df["wMAPE"].iloc[0]})

        # Log artifacts
        with aiplatform.start_execution(
            schema_title="system.ContainerExecution", display_name="evaluation"
        ) as execution:
            input_artifact = aiplatform.Artifact.create(
                schema_title="google.BQTable",
                display_name="Training dataset",
                resource_id=f"{experiment_name}-{experiment_run_name}-training-dataset",
                project=project_id,
                location=location,
                uri=f"bq://{project_id}.{bq_dataset_name}.{bq_table_prepped}",
                description="Training dataset",
                metadata={
                    "name": f"{bq_dataset_name}.{bq_table_prepped}",
                    "project_id": project_id,
                    "dataset_id": bq_dataset_name,
                    "table_id": f"{bq_dataset_name}.{bq_table_prepped}",
                },
            )
            execution.assign_input_artifacts([input_artifact])

            output_artifact = aiplatform.Artifact.create(
                schema_title="google.BQTable",
                display_name="Prediction and explainability dataset",
                resource_id=f"{experiment_name}-{experiment_run_name}-prediction-dataset",
                project=project_id,
                location=location,
                uri=f"bq://{project_id}.{bq_dataset_name}.{bq_eval_table_id}",
                description="Predictions and explainability dataset",
                metadata={
                    "name": bq_eval_table_id,
                    "project_id": project_id,
                    "dataset_id": bq_dataset_name,
                    "table_id": bq_eval_table_id,
                },
            )
            execution.assign_output_artifacts([output_artifact])
