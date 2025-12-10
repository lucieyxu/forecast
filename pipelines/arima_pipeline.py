import argparse
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List

import yaml
from arima.eval import evaluate_model
from arima.prediction import predictions
from arima.train import train_arima_model
from config import PIPELINE_BUCKET, PROJECT_ID, REGION
from common.split_train_test import split_train_test
from google.cloud import aiplatform, storage
from kfp.compiler import Compiler
from kfp.dsl import Collected, ParallelFor, pipeline
from kfp.dsl.base_component import BaseComponent
from time_split import TimeSplit


def generate_forecasting_pipeline(
    project_id: str,
    region: str,
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
    time_splits_range: List[int],
    pipeline_bucket: str,
    rolling_window_time_split_gcs_name: str,
) -> BaseComponent:
    """Generates the main forecasting pipeline that runs train/eval per fold
    and aggregates results. Assumes characteristics are pre-generated."""

    @pipeline(name=f"arima-training-pipeline-{experiment_run_name}")
    def forecasting_pipeline():
        with ParallelFor(time_splits_range) as fold:
            data_prepped_filtered_fold_n = (
                f"{bq_table_prepped}-fold-{fold}-{experiment_run_name}"
            )
            bq_model_name_fold_n = f"{bq_model_name}-fold-{fold}-{experiment_run_name}"
            experiment_run_name_fold_n = f"{experiment_run_name}-fold-{fold}"

            split_train_test_op = split_train_test(
                project_id=project_id,
                bq_dataset_name=bq_dataset_name,
                bq_table_source=bq_table_prepped,
                bq_table_prepped=data_prepped_filtered_fold_n,
                time_column=time_column,
                split_column=split_column,
                fold=fold,
                pipeline_bucket=pipeline_bucket,
                rolling_window_time_split_gcs_name=rolling_window_time_split_gcs_name,
            )

            train_model_task = (
                train_arima_model(
                    project_id=project_id,
                    bq_dataset_name=bq_dataset_name,
                    bq_table_prepped=data_prepped_filtered_fold_n,
                    bq_model_name=bq_model_name_fold_n,
                    experiment_name=experiment_name,
                    experiment_description=experiment_description,
                    experiment_run_name=experiment_run_name_fold_n,
                    model_type=model_type,
                    target_column=target_column,
                    time_column=time_column,
                    series_column=series_column,
                    split_column=split_column,
                    forecast_granularity=forecast_granularity,
                    options=options,
                    covariates=covariates,
                    fold=fold,
                    pipeline_bucket=pipeline_bucket,
                    rolling_window_time_split_gcs_name=rolling_window_time_split_gcs_name,
                )
                .set_retry(2)
                .after(split_train_test_op)
            )

            predictions_task = (
                predictions(
                    project_id=project_id,
                    bq_dataset_name=bq_dataset_name,
                    bq_table_prepped=data_prepped_filtered_fold_n,
                    bq_model_name=bq_model_name_fold_n,
                    experiment_name=experiment_name,
                    experiment_run_name=experiment_run_name_fold_n,
                    model_type=model_type,
                    time_column=time_column,
                    series_column=series_column,
                    target_column=target_column,
                    covariates=covariates,
                    forecast_granularity=forecast_granularity,
                    options=options,
                    fold=fold,
                    pipeline_bucket=pipeline_bucket,
                    rolling_window_time_split_gcs_name=rolling_window_time_split_gcs_name,
                )
                .set_retry(2)
                .after(train_model_task)
            )

            evaluate_model_task = (
                evaluate_model(
                    project_id=project_id,
                    location=region,
                    bq_dataset_name=bq_dataset_name,
                    bq_table_prepped=data_prepped_filtered_fold_n,
                    bq_model_name=bq_model_name_fold_n,
                    experiment_name=experiment_name,
                    experiment_description=experiment_description,
                    experiment_run_name=experiment_run_name_fold_n,
                    model_type=model_type,
                    time_column=time_column,
                    series_column=series_column,
                    target_column=target_column,
                    covariates=covariates,
                    forecast_granularity=forecast_granularity,
                    options=options,
                    fold=fold,
                    pipeline_bucket=pipeline_bucket,
                    rolling_window_time_split_gcs_name=rolling_window_time_split_gcs_name,
                )
                .set_retry(2)
                .after(train_model_task)
            )

    return forecasting_pipeline  # type: ignore


def create_rolling_windows(
    train_start_date,
    test_start_date: str,
    test_end_date: str,
    step_size: int,
    horizon: int,
) -> List[Dict[str, Any]]:
    """
    Generates a list of TimeSplit for a rolling window.

    Args:
        train_start_date (str): training set start date
        test_start_date (str): The initial date for the rolling window (YYYY-MM-DD).
        test_start_date (str): The final date for the rolling window (YYYY-MM-DD).
        step_size (int): The number of weeks to advance the start of each window.
        horizon (int): The fixed number of weeks for the duration of each window.

    Returns:
        List[Dict[str, Any]]: list of TimeSplit attributes as dict with train val test information.
    """
    test_start_datetime = datetime.strptime(test_start_date, "%Y-%m-%d").date()
    test_end_datetime = datetime.strptime(test_end_date, "%Y-%m-%d").date()
    window_start = test_start_datetime
    windows = []

    i = 0
    while window_start <= test_end_datetime:
        window_end = window_start + timedelta(weeks=horizon)
        time_split = TimeSplit(
            train_start_date=train_start_date,
            test_start_date=window_start.strftime("%Y-%m-%d"),
            test_end_date=window_end.strftime("%Y-%m-%d"),
            horizon=horizon,
        )
        time_split_dict = time_split.__dict__
        time_split_dict["fold"] = i

        windows.append(time_split_dict)
        window_start += timedelta(weeks=step_size)
        i += 1

    return windows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training pipeline arguments")
    parser.add_argument(
        "--experiment",
        type=str,
        help="Local path to experiment yaml to use",
    )
    args = parser.parse_args()
    with open(args.experiment, "r") as file:
        experiment_config = yaml.safe_load(file)

    experiment_name = experiment_config["Experiment"]["name"]
    horizon = experiment_config["Model"]["horizon"]
    window_step_size = experiment_config["Model"].get("window_step_size", horizon)

    time_splits: List[Dict[str, Any]] = create_rolling_windows(
        train_start_date=experiment_config["Model"]["train_date_lower"],
        test_start_date=experiment_config["Model"]["val_date_upper"],
        test_end_date=experiment_config["Model"]["test_date_upper"],
        horizon=horizon,
        step_size=window_step_size,
    )
    len_time_splits = len(time_splits)
    storage_client = storage.Client(project=PROJECT_ID)
    pipeline_bucket = storage_client.bucket(PIPELINE_BUCKET)
    json_str = json.dumps(time_splits)
    for window in time_splits:
        print(window)
    experiment_run_name = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Need to dump to json to use kfp ParallelFor otherwise ongoing bugs
    # When using the list directly
    # https://github.com/kubeflow/pipelines/issues/9366
    # https://github.com/kubeflow/pipelines/issues/9777
    blob_name = f"{experiment_name}/{experiment_run_name}/time_splits.json"
    blob = pipeline_bucket.blob(blob_name)
    blob.upload_from_string(json_str, content_type="application/json")

    print(f"Saved time splits in bucket {pipeline_bucket} at {blob_name}")
    print(f"Running pipeline with {len_time_splits} windows.")

    aiplatform.init(project=PROJECT_ID, location=REGION)
    compiler = Compiler()

    forecasting_pipeline_func = generate_forecasting_pipeline(
        project_id=PROJECT_ID,
        region=REGION,
        bq_dataset_name=experiment_config["BQ"]["dataset"],
        bq_table_prepped=experiment_config["BQ"]["table_prepped"],
        bq_model_name=experiment_config["BQ"]["model_name"],
        experiment_name=experiment_name,
        experiment_description=experiment_config["Experiment"]["description"],
        experiment_run_name=experiment_run_name,
        model_type=experiment_config["Model"]["type"],
        target_column=experiment_config["Model"]["target_col"],
        time_column=experiment_config["Model"]["time_col"],
        series_column=experiment_config["Model"]["series_col"],
        split_column=f'{experiment_config["Model"]["split_col"]}',
        forecast_granularity=experiment_config["Model"]["forecast_granularity"],
        options=experiment_config["Model"]["options"],
        covariates=(
            experiment_config["Model"]["covariate_cols"] 
            if "covariate_cols" in experiment_config["Model"] 
            else []
        ),
        time_splits_range=list(range(len_time_splits)),
        pipeline_bucket=PIPELINE_BUCKET,
        rolling_window_time_split_gcs_name=blob_name,
    )

    forecasting_pipeline_def_file = f"arima-training-pipeline-{experiment_run_name}.json"
    compiler.compile(
        pipeline_func=forecasting_pipeline_func,
        package_path=forecasting_pipeline_def_file,
    )

    training_job = aiplatform.PipelineJob(
        display_name=f"{experiment_name}-forecasting-pipeline-{experiment_run_name}",
        template_path=forecasting_pipeline_def_file,
        pipeline_root=f"gs://{PIPELINE_BUCKET}/pipeline_root/{experiment_name}-arima-pipeline-{experiment_run_name}",
        enable_caching=True,
    )
    print(
        f"Submitting forecasting pipeline with run name {experiment_run_name}"
    )
    training_job.run(sync=False)
    print("Forecasting pipeline submitted.")
