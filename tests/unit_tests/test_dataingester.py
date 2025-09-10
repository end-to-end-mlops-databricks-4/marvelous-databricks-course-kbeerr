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
    # Check float columns
    for col in cast_as_float or []:
        assert col in sdf.columns
        field_type = [f.dataType.typeName() for f in sdf.schema.fields if f.name == col][0]
        assert field_type == "float"

    # Check string columns
    for col in cast_as_string or []:
        assert col in sdf.columns
        field_type = [f.dataType.typeName() for f in sdf.schema.fields if f.name == col][0]
        assert field_type == "string"

    # Check remaining columns
    remaining_cols = set(sdf.columns) - set(cast_as_float or []) - set(cast_as_string or [])
    for col in remaining_cols:
        assert col in sdf.columns
        field_type = [f.dataType.typeName() for f in sdf.schema.fields if f.name == col][0]
        expected_type = "float" if cast_remaining_as == FloatType else "string"
        assert field_type == expected_type

    # Check datetime column if applicable
    if datetime_column:
        assert "datetime" in sdf.columns
        field_type = [f.dataType.typeName() for f in sdf.schema.fields if f.name == "datetime"][0]
        assert field_type == "timestamp"
