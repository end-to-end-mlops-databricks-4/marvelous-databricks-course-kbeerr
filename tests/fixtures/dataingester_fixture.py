"""Fixtures for tests."""

import pandas as pd
import pytest
from loguru import logger
from pyspark.sql import SparkSession

from zonnedael import PROJECT_DIR
from zonnedael.config import ProjectConfig, Tags
from tests.unit_tests.spark_config import spark_config
from zonnedael.ingest.data_ingester import DataIngester


@pytest.fixture(scope="session")
def spark_session() -> SparkSession:
    """Create and return a SparkSession for testing.

    This fixture creates a SparkSession with the specified configuration and returns it for use in tests.
    """
    # One way
    # spark = SparkSession.builder.getOrCreate()  # noqa
    # Alternative way - better
    spark = (
        SparkSession.builder.master(spark_config.master)
        .appName(spark_config.app_name)
        .config("spark.executor.cores", spark_config.spark_executor_cores)
        .config("spark.executor.instances", spark_config.spark_executor_instances)
        .config("spark.sql.shuffle.partitions", spark_config.spark_sql_shuffle_partitions)
        .config("spark.driver.bindAddress", spark_config.spark_driver_bindAddress)
        .getOrCreate()
    )

    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def config() -> ProjectConfig:
    """Load and return the project configuration.

    This fixture reads the project configuration from a YAML file and returns a ProjectConfig object.

    :return: The loaded project configuration
    """
    config_file_path = (PROJECT_DIR / "project_config.yml").resolve()
    logger.info(f"Current config file path: {config_file_path.as_posix()}")
    config = ProjectConfig.from_yaml(config_file_path.as_posix())
    return config

@pytest.fixture(scope="function")
def ingester(config, spark_session):
    """Create a DataIngester instance for tests."""
    base_path = str(config.project_dir / "tests" / "test_data")
    return DataIngester(base_path=base_path, config=config, spark=spark_session)