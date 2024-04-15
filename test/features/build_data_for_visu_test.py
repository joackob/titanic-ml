import pandas as pd
import pathlib as pl

from src.features.build_data_for_visu import build_data_for_visu

data_raw_path = pl.Path("data/raw/train.csv")
data_raw_path.resolve()
data_raw = pd.read_csv(filepath_or_buffer=data_raw_path)

data_processed_for_visu = build_data_for_visu(data_raw)


def test_data_transformed_not_lose_rows():
    assert data_processed_for_visu.shape[0] == data_raw.shape[0]


def test_can_access_to_fields_with_long_name():
    assert "#Hermanos o conyuges a bordo" in data_processed_for_visu.columns


def test_data_transformed_have_eleven_columns():
    assert data_processed_for_visu.shape[1] == 12


def test_data_transformed_have_column_age_range():
    assert "Rango etario" in data_processed_for_visu.columns
