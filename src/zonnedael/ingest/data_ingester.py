"""Data ingester for Zonnedael dataset."""
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F
from typing import Optional, Union, List
from zonnedael.config import ProjectConfig

class DataIngester:
    """A class to ingest data into a Spark DataFrame and upload to Unity Catalog.
    """

    def __init__(self, base_path: str, config: ProjectConfig, spark: SparkSession):
        """Initialize the DataIngester with file path and name.

        :param path: The directory path where the data file is located
        :param filename: The name of the data file to be ingested
        """
        self.base_path = base_path
        self.config = config
        self.spark = spark
        self.sdf = None
    
    def _read_data(self, filename: str, delimiter: str = ";") -> None:
        """Read the data file into a Spark DataFrame.

        :param filename: The name of the data file to be ingested
        :param delimiter: The delimiter used in the CSV file (default is ';')
        """
        self.sdf = self.spark.read.options(
            delimiter=delimiter,
            header=True,
            inferSchema=True
        ).csv(self.base_path + "/" + filename)

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
            datetime_column: Optional[str] = None,
            datetime_format: Optional[str] = None,
            cast_as_float: List[str] = [],
            cast_as_string: List[str] = [],
            cast_remaining_as: Union[FloatType, StringType] = StringType
            ) -> None:
        """Cast columns to appropriate data types.

        :param datetime_column: The name of the datetime column to be converted
        :param datetime_format: The format string for parsing the datetime column
        :param cast_remaining_as: The data type to cast remaining columns (default is StringType)
        """
        if datetime_column:
            self.sdf = self.sdf.withColumn(datetime_column, F.to_timestamp(F.col(datetime_column), datetime_format))
        for c in self.sdf.columns:
            if c == datetime_column:
                continue
            if c in cast_as_float:
                self.sdf = self.sdf.withColumn(c, F.col(c).cast(FloatType()))
            elif c in cast_as_string:
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
            missing_value_indicator: str = "#WAARDE!",
            datetime_column: Optional[str] = None,
            datetime_format: Optional[str] = None,
            cast_as_float: List[str] = [],
            cast_as_string: List[str] = [],
            cast_remaining_as: Union[FloatType, StringType] = StringType,
            mode: str = "overwrite"
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
        """
        self._read_data(filename, delimiter)
        self._clean_column_names()
        self._replace_missing_values(missing_value_indicator)
        self._cast_column_types(
            datetime_column,
            datetime_format,
            cast_as_float,
            cast_as_string,
            cast_remaining_as
        )
        self._drop_all_null_rows()
        self._upload_to_unity_catalog(table_name, mode)