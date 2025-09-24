"""Data ingester for Zonnedael dataset."""

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, StringType

from zonnedael.config import ProjectConfig


class DataIngester:
    """A class to ingest data into a Spark DataFrame and upload to Unity Catalog."""

    def __init__(self, base_path: str, config: ProjectConfig, spark: SparkSession) -> None:
        """Initialize the DataIngester with file path and name.

        :param path: The directory path where the data file is located
        :param filename: The name of the data file to be ingested
        """
        self.base_path = base_path
        self.config = config
        self.spark = spark
        self.sdf = None

    def _read_data(self, filename: str, delimiter: str = ";", skip_rows: int = 0) -> None:
        """Read the data file into a Spark DataFrame.

        :param filename: The name of the data file to be ingested
        :param delimiter: The delimiter used in the CSV file (default is ';')
        :param skip_rows: Number of rows to skip before the header (default is 0)
        """
        options = {
            "delimiter": delimiter,
            "header": True,
            "inferSchema": True,
            "ignoreLeadingWhiteSpace": True,
            "ignoreTrailingWhiteSpace": True,
            "comment": "#",
        }
        if skip_rows > 0:
            options["skipRows"] = skip_rows
        self.sdf = self.spark.read.options(**options).csv(self.base_path + "/" + filename)

    def _clean_column_names(self) -> None:
        """Clean column names by replacing spaces with underscores and converting to lowercase."""
        for old_name in self.sdf.columns:
            new_name = old_name.replace(" ", "_").lower()
            self.sdf = self.sdf.withColumnRenamed(old_name, new_name)

    def _replace_missing_values(self, missing_value_indicator: str = "#WAARDE!") -> None:
        """Replace specified missing value indicators with None (null) in all columns.

        :param missing_value_indicator: The string that indicates a missing value (default is '#WAARDE!')
        """
        for c in self.sdf.columns:
            self.sdf = self.sdf.withColumn(c, F.regexp_replace(F.col(c), missing_value_indicator, ""))

    def _cast_column_types(
        self,
        datetime_column: str | list[str] | None = None,
        datetime_format: str | None = None,
        cast_as_float: list[str] | None = None,
        cast_as_string: list[str] | None = None,
        cast_remaining_as: type = StringType,
    ) -> None:
        """Cast columns to appropriate data types.

        :param datetime_column: Name(s) of column(s) to be combined into a datetime.
            - None: do nothing
            - str: treat as single datetime column
            - list[str]: concatenate into one column. Make sure the first element is the date part and the second the time part.
        :param datetime_format: The format string for parsing the datetime column. If hour is mixed "H" and "HH", use "HH". Function will pad zeroes at hours <10.
            (e.g. "yyyyMMddHH")
        :param cast_remaining_as: Type to cast remaining columns (default StringType).
        """
        if cast_as_float is None:
            cast_as_float = []
        if cast_as_string is None:
            cast_as_string = []
        # Handle datetime column(s)
        combined_dt_col = None
        if datetime_column:
            if isinstance(datetime_column, str):
                # Single datetime column
                self.sdf = self.sdf.withColumn(datetime_column, F.to_timestamp(F.col(datetime_column), datetime_format))
            elif isinstance(datetime_column, list):
                # This is KMNI-specific handling for separate date and hour columns
                # But I am not going to modularize this more as this is a one-off ingestion
                # For a course project, sorry

                # Pad and subtract 1 hour IF hour is 1-24 instead of 0-23
                # This is the case in the KNMI data where HH=24 means 00:00 of the next day
                # Check if HH column contains "24"
                distinct_hours = [row[0] for row in self.sdf.select("HH").distinct().collect()]

                if 24 in [int(h) for h in distinct_hours if h is not None]:
                    self.sdf = self.sdf.withColumn(
                        datetime_column[1], (F.col(datetime_column[1]).cast("int") - 1).cast("int")
                    )
                # Fix HH=0..23 as 2-digit strings
                self.sdf = self.sdf.withColumn(
                    datetime_column[1], F.lpad(F.col(datetime_column[1]).cast("string"), 2, "0")
                )

                # Combine multiple columns into one string
                combined_dt_col = "datetime"
                self.sdf = self.sdf.withColumn(combined_dt_col, F.concat_ws("", *[F.col(c) for c in datetime_column]))
                self.sdf = self.sdf.withColumn(combined_dt_col, F.to_timestamp(F.col(combined_dt_col), datetime_format))

                # Drop original datetime parts
                for c in datetime_column:
                    self.sdf = self.sdf.drop(c)

        for c in self.sdf.columns:
            if isinstance(datetime_column, str) and c == datetime_column:
                continue
            # keep datetime parts as String if datetime_column is a list
            elif isinstance(datetime_column, list) and c in datetime_column:
                self.sdf = self.sdf.withColumn(c, F.col(c).cast(StringType()))
            elif c == combined_dt_col:
                continue
            elif c in cast_as_float:
                self.sdf = self.sdf.withColumn(c, F.col(c).cast(FloatType()))
            elif c in cast_as_string or c in cast_as_string:
                self.sdf = self.sdf.withColumn(c, F.col(c).cast(StringType()))
            else:
                self.sdf = self.sdf.withColumn(c, F.col(c).cast(cast_remaining_as()))

    def _drop_all_null_rows(self) -> None:
        """Remove rows where all columns are null."""
        self.sdf = self.sdf.na.drop(how="all")

    def _upload_to_unity_catalog(self, table_name: str, mode: str = "overwrite") -> None:
        """Upload the DataFrame to Unity Catalog.

        :param table_name: The name of the table in Unity Catalog
        :param mode: The write mode (default is 'overwrite')
        """
        catalog = self.config.catalog_name
        schema = self.config.schema_name
        self.sdf.write.mode(mode).option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.{table_name}")

    def ingest(
        self,
        filename: str,
        table_name: str,
        delimiter: str = ";",
        missing_value_indicator: str | None = None,
        datetime_column: str | None = None,
        datetime_format: str | None = None,
        cast_as_float: list[str] | None = None,
        cast_as_string: list[str] | None = None,
        cast_remaining_as: type = StringType,
        mode: str = "overwrite",
        skip_rows: int = 0,
        test_mode: bool = False,
    ) -> None:
        """Full pipeline to read, clean, and upload data.

        :param filename: The name of the data file to be ingested
        :param table_name: The name of the table in Unity Catalog
        :param delimiter: The delimiter used in the CSV file (default is ';')
        :param missing_value_indicator: The string that indicates a missing value (default is '#WAARDE!')
        :param datetime_column: The name of the datetime column to be converted
        :param datetime_format: The format string for parsing the datetime column
        :param cast_as_float: List of columns to cast as FloatType
        :param cast_as_string: List of columns to cast as StringType
        :param cast_remaining_as: The data type to cast remaining columns (default is StringType)
        :param mode: The write mode (default is 'overwrite')
        :param skip_rows: Number of rows to skip before the header (default is 0)
        :param test_mode: If True, skip uploading to Unity Catalog (default is False)
        """
        self._read_data(filename, delimiter, skip_rows)
        self._clean_column_names()
        if missing_value_indicator:
            self._replace_missing_values(missing_value_indicator)
        self._cast_column_types(datetime_column, datetime_format, cast_as_float, cast_as_string, cast_remaining_as)
        self._drop_all_null_rows()
        if not test_mode:
            self._upload_to_unity_catalog(table_name, mode)
