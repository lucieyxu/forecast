from typing import Dict, List

from kfp.dsl import component


@component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-bigquery",
        "google-cloud-storage",
        "pandas",
        "db-dtypes",
        "tabulate",
    ],
)
def split_train_test(
    project_id: str,
    bq_dataset_name: str,
    bq_table_source: str,
    bq_table_prepped: str,
    time_column: str,
    split_column: str,
    fold: int,
    pipeline_bucket: str,
    rolling_window_time_split_gcs_name: str,
) -> None:
    """Split train val test sets.

    Args:
        project_id (str): project ID
        bq_dataset_name (str): BigQuery dataset name
        bq_table_source (str): BigQuery table to query from
        bq_table_prepped (str): BigQuery table to create or replace
        time_column (str): Timestamp column to use for forecast
        split_column (str): Column containing categories for train val test
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
    train_date_upper = time_split["val_start_date"]
    val_date_upper = time_split["test_start_date"]
    test_date_upper = time_split["test_end_date"]

    bq = bigquery.Client(project=project_id)

    # Get the list of columns in the source table to check if split column is already present
    table_ref = bigquery.TableReference.from_string(
        f"{project_id}.{bq_dataset_name}.{bq_table_source}"
    )
    table = bq.get_table(table_ref)
    columns = [field.name for field in table.schema]
    exept_col_sql_str = ""
    if split_column in columns:
        exept_col_sql_str = f"EXCEPT(`{split_column}`)"

    query = f"""
        CREATE OR REPLACE TABLE `{project_id}.{bq_dataset_name}.{bq_table_prepped}` AS
        SELECT * {exept_col_sql_str},
            CASE
                WHEN {time_column} < '{train_date_upper}' THEN 'TRAIN'
                WHEN {time_column} >= '{train_date_upper}' AND {time_column} < '{val_date_upper}' THEN 'VALIDATE'
                WHEN {time_column} >= '{val_date_upper}' AND {time_column} < '{test_date_upper}' THEN 'TEST'
                ELSE NULL
        END AS {split_column}
        FROM `{project_id}.{bq_dataset_name}.{bq_table_source}`
        ORDER BY {time_column}
    """
    logger.info(query)
    job = bq.query(query)
    job.result()
    logger.info(
        f"Split train val test data in {(job.ended-job.started).total_seconds()} s"
    )  # type: ignore

    query = f"""
        SELECT {split_column}, COUNT(*)
        FROM `{project_id}.{bq_dataset_name}.{bq_table_prepped}`
        GROUP BY 1
    """
    job = bq.query(query)
    count_df = job.to_dataframe()
    logger.info(count_df.to_markdown(index=False, numalign="left", stralign="left"))
