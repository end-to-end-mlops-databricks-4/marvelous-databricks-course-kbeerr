# Databricks notebook source

# MAGIC %pip install -e ..
# MAGIC %restart_python
# COMMAND ----------
from pathlib import Path
import sys, os
sys.path.append(str(Path.cwd().parent / 'src'))
# COMMAND ----------
from loguru import logger
import yaml
# import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *
# import pyspark.sql.functions as F
# import pandas as pd

from zonnedael.config import ProjectConfig
from zonnedael.ingest.data_ingester import DataIngester

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
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
    mode="overwrite"
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
    mode="overwrite"
)
logger.info("Ingestion of 'Zonnedael - slimme meter dataset - 2013 - klanttypering.csv' completed.")