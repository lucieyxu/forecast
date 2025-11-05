from typing import Dict, List

from kfp.dsl import component


@component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-aiplatform",
        "google-cloud-storage",
    ],
)
def train_model(
    project_id: str,
    region: str,
    bq_dataset_name: str,
    bq_table_prepped: str,
    experiment_name: str,
    experiment_description: str,
    experiment_run_name: str,
    model_type: str,
    target_column: str,
    time_column: str,
    series_column: str,
    split_column: str,
    optimization_objective: str,
    attributes_columns: List[str],
    covariates_columns_known: List[str],
    covariates_columns_unknown: List[str],
    forecast_granularity: str,
    context_window: int,
    holiday_regions: List[str],
    budget_milli_node_hours: int,
    fold: int,
    pipeline_bucket: str,
    rolling_window_time_split_gcs_name: str,
) -> None:
    """AutoML training.

    Args:
        project_id (str): project ID
        region (str): VAI Experiment region
        bq_dataset_name (str): BigQuery dataset name
        bq_table_prepped (str): BigQuery table to create or replace
        experiment_name (str): VertexAI experiment name
        experiment_description (str): VertexAI experiment description
        experiment_run_name (str): VertexAI experiment run name
        model_type (str): Model type. AutoMLForecastingTrainingJob or TimeSeriesDenseEncoderForecastingTrainingJob
        target_column (str): Column with value to forecast
        time_column (str): Timestamp column to use for forecast
        series_column (str): Column to use to get the forecast granularity. Each unique ID will be a time series
        split_column (str): Column containing categories for train val test
        optimization_objective (str): objective to minimize
        attributes_columns (List[str]): Attributes of time series to use in training (static)
        covariates_columns_known (List[str]): Covariates known at time of prediction (Promotions for example)
        covariates_columns_unknown (List[str]): Covariates unknown at time of prediction (stock for example)
        forecast_granularity (str): DAY or WEEK
        context_window (int): Training context window
        holiday_regions (List[str]): List of holidays to use
        budget_milli_node_hours (int): Compute budget in milli node hours
        fold (int): when using rolling window training pipeline
        pipeline_bucket (str): pipeline bucket
        rolling_window_time_split_gcs_name (str): time splits gcs name
    """
    import json
    import logging
    import random
    import time

    from google.api_core.exceptions import ResourceExhausted
    from google.cloud import aiplatform, storage
    from google.cloud.aiplatform import gapic as aip_gapic
    from google.cloud.aiplatform.compat.services import pipeline_service_client

    logger = logging.getLogger(__name__)

    st = time.perf_counter()
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(pipeline_bucket)
    blob = bucket.blob(rolling_window_time_split_gcs_name)
    time_split = json.loads(blob.download_as_string())[fold]
    forecast_horizon_length = time_split["test_length"]

    aiplatform.init(project=project_id, location=region)

    dataset = aiplatform.TimeSeriesDataset.create(
        display_name=f"{experiment_name}{experiment_run_name}",
        bq_source=f"bq://{project_id}.{bq_dataset_name}.{bq_table_prepped}",
        labels={
            "run_id": experiment_run_name,
            "experiment": experiment_name,
            "fold": str(fold),
        },
    )

    logger.info(f"Created Dataset: {dataset.display_name}")

    column_specs = dict.fromkeys(
        [target_column, time_column]
        + attributes_columns
        + covariates_columns_known
        + covariates_columns_unknown,
        "auto",
    )
    logger.info(f"Feature transformation: {column_specs}")

    # Using retry with Backoff to prevent google.api_core.exceptions.ResourceExhausted: 429 errors
    retry_count = 0
    backoff_time = 30
    max_retries = 5
    display_name = f"job_{experiment_run_name}"
    while retry_count <= max_retries:
        try:
            forecasting_job: aiplatform.VertexAiStatefulResource
            if model_type == "AutoMLForecastingTrainingJob":
                forecasting_job = aiplatform.AutoMLForecastingTrainingJob(
                    display_name=display_name,
                    optimization_objective=optimization_objective,
                    column_specs=column_specs,
                    labels={
                        "run_id": experiment_run_name,
                        "experiment": experiment_name,
                        "fold": str(fold),
                        "model_type": "l2l",
                        "context_window": str(context_window),
                    },
                )
            elif model_type == "TimeSeriesDenseEncoderForecastingTrainingJob":
                forecasting_job = (
                    aiplatform.TimeSeriesDenseEncoderForecastingTrainingJob(
                        display_name=display_name,
                        optimization_objective=optimization_objective,
                        column_specs=column_specs,
                        labels={
                            "run_id": experiment_run_name,
                            "experiment": experiment_name,
                            "fold": str(fold),
                            "model_type": "tide",
                            "context_window": str(context_window),
                        },
                    )
                )
            else:
                raise ValueError(f"Model not supported {model_type}")

            model_id = (
                f"model_{display_name}"  # Needs to be different from display_name
            )

            logger.info(f"Starting training {display_name}")
            logger.info(f"Model ID: {model_id}")
            forecast = forecasting_job.run(
                # data parameters
                dataset=dataset,
                target_column=target_column,
                time_column=time_column,
                time_series_identifier_column=series_column,
                time_series_attribute_columns=attributes_columns,
                unavailable_at_forecast_columns=[target_column]
                + covariates_columns_unknown,
                available_at_forecast_columns=[time_column] + covariates_columns_known,
                predefined_split_column_name=split_column,
                # weight_column="", # Use if some rows are more important than others
                # forecast parameters
                forecast_horizon=forecast_horizon_length,
                data_granularity_unit=forecast_granularity,
                data_granularity_count=1,
                context_window=context_window,
                holiday_regions=holiday_regions,
                # quantiles=[0.1, 0.5, 0.9], # Need to use with loss=minimize-quantile-loss
                # hierarchy_group_columns=["Brand"],
                # hierarchy_group_total_weight=10.0,  # weight of the loss for predictions aggregated over time series in the same hierarchy group.
                # hierarchy_temporal_total_weight = 2.0, # weight of the loss for predictions aggregated over the horizon for a single time series.
                # hierarchy_group_temporal_total_weight = 1.0, # weight of the loss for predictions aggregated over both the horizon and time series in the same hierarchy group.
                # output parameters
                export_evaluated_data_items=True,
                export_evaluated_data_items_bigquery_destination_uri=f"bq://{project_id}.{bq_dataset_name}.{experiment_name}_{experiment_run_name}_eval",
                export_evaluated_data_items_override_destination=True,
                # running parameters
                validation_options="fail-pipeline",
                budget_milli_node_hours=budget_milli_node_hours,
                # model parameters
                model_display_name=f"{experiment_name}_{experiment_run_name}",
                model_labels={
                    "run_id": experiment_run_name,
                    "experiment": experiment_name,
                },
                model_id=model_id,
                parent_model="",  # Always train from scratch. Will need to change for retraining
                is_default_version=True,
                # session parameters: False means continue in local session, True waits and logs progress
                sync=True,
            )
            train_time = time.perf_counter() - st

            logger.info(f"Training job display name: {forecast.display_name}")
            logger.info(f"Training job resource name: {forecast.resource_name}")
            logger.info(f"Training job ran during {train_time}")

            aiplatform.init(
                project=project_id,
                location=region,  # We use the same global region for experiment logging
                experiment=experiment_name,
                experiment_description=experiment_description,
            )
            with aiplatform.start_run(experiment_run_name) as run:
                params_to_log = {
                    "model_type": model_type,
                    "horizon": str(forecast_horizon_length),
                    "data_frequency": forecast_granularity,
                    "model_id": display_name,
                    "time_series_attribute_columns": str(attributes_columns),
                    "unavailable_at_forecast_columns": str(covariates_columns_unknown),
                    "available_at_forecast_columns": str(covariates_columns_known),
                    "holiday_regions": str(holiday_regions),
                    "budget_milli_node_hours": budget_milli_node_hours,
                }

                logger.info(
                    f"Logging parameters to Vertex AI experiment {experiment_name} {experiment_run_name}: {params_to_log}"
                )
                run.log_params(params_to_log)
                run.log_metrics({"train_time": train_time})
                logger.info("Done logging.")
            return
        except ResourceExhausted as re:
            if retry_count < max_retries:
                retry_count += 1
                logger.warning(
                    f"ResourceExhausted error encountered ({re}). Retrying in {backoff_time} seconds (attempt {retry_count}/{max_retries})..."
                )
                time.sleep(backoff_time + random.randint(0, 30))
                backoff_time *= 2
            else:
                logger.error(f"Max retries reached: {re}")
                raise re
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise e
