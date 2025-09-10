"""Unit tests for DataIngester."""

import pytest
from pyspark.sql import DataFrame
from pyspark.sql.types import FloatType, StringType

from zonnedael.ingest.data_ingester import DataIngester


@pytest.mark.parametrize(
    "filename,delimiter,missing_value_indicator,datetime_column,datetime_format,cast_as_float,cast_as_string,cast_remaining_as",
    [
        ("zonnedael_levering.csv", ";", "#WAARDE!", "datetime", "d-M-yyyy H:mm", None, None, FloatType),
        ("zonnedael_klanttypering.csv", ";", "#WAARDE!", None, None, None, None, StringType),
        ("knmi_weatherdata.txt", ",", None, ["yyyymmdd", "hh"], "yyyyMMddHH", None, None, FloatType),
    ],
)
def test_ingest_pipeline(
    ingester: DataIngester,
    filename: str,
    delimiter: str,
    missing_value_indicator: str | None,
    datetime_column: str | list[str] | None,
    datetime_format: str | None,
    cast_as_float: list[str] | None,
    cast_as_string: list[str] | None,
    cast_remaining_as: type,
) -> None:
    """Test full ingest pipeline for each CSV file in test mode."""
    ingester.ingest(
        filename=filename,
        table_name="test_table",
        delimiter=delimiter,
        missing_value_indicator=missing_value_indicator,
        datetime_column=datetime_column,
        datetime_format=datetime_format,
        cast_as_float=cast_as_float,
        cast_as_string=cast_as_string,
        cast_remaining_as=cast_remaining_as,
        mode="overwrite",
        test_mode=True,
    )

    sdf: DataFrame = ingester.sdf
    assert sdf.count() > 0  # Ensure rows were read
    for col in sdf.columns:
        field_type = next(f.dataType.typeName() for f in sdf.schema.fields if f.name == col)

        if col in (cast_as_float or []):
            assert field_type == "float", f"{col} expected float, got {field_type}"
        elif col in (cast_as_string or []):
            assert field_type == "string", f"{col} expected string, got {field_type}"
        elif datetime_column and (
            col == datetime_column if isinstance(datetime_column, str) else col in datetime_column
        ):
            assert field_type == "timestamp", f"{col} expected timestamp, got {field_type}"
        else:
            expected_type = "float" if cast_remaining_as == FloatType else "string"
            assert field_type == expected_type, f"{col} expected {expected_type}, got {field_type}"

    
