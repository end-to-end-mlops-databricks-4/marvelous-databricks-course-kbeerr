# Databricks notebook source

# MAGIC %pip install -e ..
# MAGIC %restart_python
# COMMAND ----------
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent / "src"))
# COMMAND ----------
import yaml
from loguru import logger

# import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, StringType

# import pyspark.sql.functions as F
# import pandas as pd
from zonnedael.config import ProjectConfig
from zonnedael.ingest.data_ingester import DataIngester
from zonnedael.preprocess.preprocess import Preprocessing

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
logger.info("Spark session created.")
# COMMAND ----------
base_path = "dbfs:/Workspace/Users/kabir.razack@gmail.com/data"
data_ingester = DataIngester(base_path=base_path, config=config, spark=spark)
logger.info("DataIngester initialized.")
# COMMAND ----------
# Ingest "Zonnedael - slimme meter dataset - 2013 - Levering.csv"
data_ingester.ingest(
    filename="Zonnedael - slimme meter dataset - 2013 - Levering.csv",
    table_name="zonnedael_levering",
    delimiter=";",
    missing_value_indicator="#WAARDE!",
    datetime_column="datetime",
    datetime_format="d-M-yyyy H:mm",
    cast_remaining_as=FloatType,
    mode="overwrite",
)
logger.info("Ingestion of 'Zonnedael - slimme meter dataset - 2013 - Levering.csv' completed.")
# COMMAND ----------
# Ingest "Zonnedael - slimme meter dataset - 2013 - klanttypering.csv"
data_ingester.ingest(
    filename="Zonnedael - slimme meter dataset - 2013 - klanttypering.csv",
    table_name="zonnedael_klanttypering",
    delimiter=";",
    missing_value_indicator="#WAARDE!",
    cast_remaining_as=StringType,
    mode="overwrite",
)
logger.info("Ingestion of 'Zonnedael - slimme meter dataset - 2013 - klanttypering.csv' completed.")
# COMMAND ----------
# Ingest KNMI weather data
data_ingester.ingest(
    filename="uurgeg_273_2011-2020.txt",
    table_name="knmi_weather_data_2011_2020",
    delimiter=",",
    missing_value_indicator=None,
    datetime_column=["yyyymmdd", "hh"],
    datetime_format="yyyyMMddHH",
    cast_remaining_as=FloatType,
    mode="overwrite",
)
logger.info("Ingestion of 'uurgeg_273_2011-2020.txt' completed.")
# COMMAND ----------
preprocessor = Preprocessing(config=config, spark=spark)
logger.info("Preprocessing initialized.")
# COMMAND ----------
# Preprocess targets
preprocessor.preprocess_targets()
logger.info("Preprocessing of targets completed.")
# COMMAND ----------
# Preprocess features
preprocessor.preprocess_features()
logger.info("Preprocessing of features completed.")
# COMMAND ----------
