"""Preprocessing for Zonnedael targets."""

from __future__ import annotations

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession

from zonnedael.config import ProjectConfig


class Preprocessing:
    """Preprocess Zonnedael target data."""

    def __init__(self, config: ProjectConfig, spark: SparkSession) -> None:
        """Initialize the Preprocessing class.

        :param config: Project configuration object containing catalog and schema info.
        :param spark: SparkSession to use for reading and writing data.
        """
        self.config = config
        self.spark = spark

    def preprocess_targets(self) -> None:
        """Preprocess target data.

        Drop unnecessary columns, unpivot customer columns,
        write to `targets` table in Unity Catalog, and return the preprocessed DataFrame.
        """
        catalog: str = self.config.catalog_name
        schema: str = self.config.schema_name
        output_table: str = f"{catalog}.{schema}.targets"

        # Read source table
        sdf: DataFrame = self.spark.table(f"{catalog}.{schema}.zonnedael_levering")

        # Drop unnecessary columns
        columns_to_drop: list[str] = ["som", "leverende_klanten", "niet_leverenden"]
        sdf = sdf.drop(*columns_to_drop)

        # Get all columns from the Spark DataFrame
        all_columns = sdf.columns

        # Filter columns containing 'klant'
        customer_columns = [col for col in all_columns if "klant" in col]

        sdf_unpivoted = (
            sdf.unpivot(
                ids="datetime", values=customer_columns, variableColumnName="customer_id", valueColumnName="target"
            )
            .withColumn("target_name", F.lit("electricity_consumption_kwh"))
            .withColumn("customer_id", F.regexp_replace(F.col("customer_id"), "klant_", "").cast("int"))
        )

        # Write to Unity Catalog table
        sdf_unpivoted.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)

    def preprocess_features(self) -> None:
        """Preprocess feature data.

        Join KNMI weather data with customer typology data,
        add additional datetime features, and write to `features` table in Unity Catalog.
        """
        catalog: str = self.config.catalog_name
        schema: str = self.config.schema_name
        output_table: str = f"{catalog}.{schema}.features"

        # Read source tables
        sdf_knmi: DataFrame = self.spark.table(f"{catalog}.{schema}.knmi_weather_data_2011_2020")
        sdf_klanttypering: DataFrame = self.spark.table(f"{catalog}.{schema}.zonnedael_klanttypering")

        # Trim trailing and leading spaces
        # Replace space dash space (" - ") with underscore ("_")
        # Replace spaces with underscores ("_")
        # Convert to lowercase
        for col in ["klant", "woning_type", "bouwjaar", "gezinssituatie"]:
            sdf_klanttypering = sdf_klanttypering.withColumn(col, F.trim(F.col(col)))
            sdf_klanttypering = sdf_klanttypering.withColumn(col, F.regexp_replace(F.col(col), " - ", "_"))
            sdf_klanttypering = sdf_klanttypering.withColumn(col, F.regexp_replace(F.col(col), " ", "_"))
            sdf_klanttypering = sdf_klanttypering.withColumn(col, F.lower(F.col(col)))

        # Transform klanttypering 'klant' column to int
        sdf_klanttypering = sdf_klanttypering.withColumn(
            "customer_id", F.regexp_replace(F.col("klant"), "klant_", "").cast("int")
        ).drop("klant")
        # Impute woning_type, bouwjaar, and gezinssituatie missing values with 'unknown'
        sdf_klanttypering = sdf_klanttypering.fillna(
            {"woning_type": "unknown", "bouwjaar": "unknown", "gezinssituatie": "unknown"}
        )

        # Join kmni weather data to klanttypering via a cross join, as there is no key to join on
        sdf_features = sdf_knmi.crossJoin(sdf_klanttypering)

        # Add additional datetime features
        sdf_features = (
            sdf_features.withColumn("year", F.year(F.col("datetime")).cast("int"))
            .withColumn("month", F.month(F.col("datetime")).cast("int"))
            .withColumn("day", F.dayofmonth(F.col("datetime")).cast("int"))
            .withColumn("hour", F.hour(F.col("datetime")).cast("int"))
            .withColumn("day_of_week", F.date_format(F.col("datetime"), "E"))
            .withColumn("is_weekend", F.when(F.col("day_of_week").isin(["Sat", "Sun"]), 1).otherwise(0).cast("int"))
        )

        # Add column descriptions as metadata
        column_descriptions = {
            "datetime": "Forward looking Timestamp of the observation",
            "woning_type": "Type of dwelling",
            "bouwjaar": "Year of construction of the dwelling",
            "gezinssituatie": "Family situation (parents, children, single, etc.)",
            "customer_id": "Customer identifier",
            "stn": "Station number (273 = Marknesse)",
            "dd": "Mean wind direction (in degrees) during the 10-minute period preceding the time of observation (360=north, 90=east, 180=south, 270=west, 0=calm, 990=variable)",
            "fh": "Hourly mean wind speed (in 0.1 m/s)",
            "ff": "Mean wind speed (in 0.1 m/s) during the 10-minute period preceding the time of observation",
            "fx": "Maximum wind gust (in 0.1 m/s) during the hourly division",
            "t": "Temperature (in 0.1 degrees Celsius) at 1.50 m at the end of the hour of observation",
            "t10n": "Minimum temperature (in 0.1 degrees Celsius) at 0.1 m in the preceding 6-hour period",
            "td": "Dew point temperature (in 0.1 degrees Celsius) at 1.50 m at the time of observation",
            "sq": "Sunshine duration (in 0.1 hour) during the hourly division, calculated from global radiation (-1 for <0.05 hour)",
            "q": "Global radiation (in J/cm2) during the hourly division",
            "dr": "Precipitation duration (in 0.1 hour) during the hourly division",
            "rh": "Hourly precipitation amount (in 0.1 mm) (-1 for <0.05 mm)",
            "p": "Air pressure (in 0.1 hPa) reduced to mean sea level, at the time of observation",
            "vv": "Horizontal visibility at the time of observation (0=less than 100m, 1=100-200m,..., 89=more than 70km)",
            "n": "Cloud cover (in octants) at the time of observation (9=sky invisible)",
            "u": "Relative atmospheric humidity (in percent) at 1.50 m at the time of observation",
            "ww": "Present weather code (00-99), description for the hourly division",
            "ix": "Indicator present weather code (1=manned and recorded, 2,3=manned and omitted, 4=automatically recorded, 5,6=automatically omitted, 7=automatically set)",
            "m": "Fog (0=no occurrence, 1=occurred during the preceding hour and/or at the time of observation)",
            "r": "Rainfall (0=no occurrence, 1=occurred during the preceding hour and/or at the time of observation)",
            "s": "Snow (0=no occurrence, 1=occurred during the preceding hour and/or at the time of observation)",
            "o": "Thunder (0=no occurrence, 1=occurred during the preceding hour and/or at the time of observation)",
            "y": "Ice formation (0=no occurrence, 1=occurred during the preceding hour and/or at the time of observation)",
            "year": "Year extracted from the datetime column",
            "month": "Month extracted from the datetime column",
            "day": "Day of the month extracted from the datetime column",
            "hour": "Hour of the day extracted from the datetime column",
            "day_of_week": "Day of the week extracted from the datetime column",
            "is_weekend": "Indicator whether the day is a weekend (1) or not (0)",
        }
        # Write to Unity Catalog table
        sdf_features.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)
        # Add column descriptions as metadata
        for col_name, desc in column_descriptions.items():
            self.spark.sql(f"ALTER TABLE {output_table} ALTER COLUMN {col_name} COMMENT '{desc}'")
