import subprocess
import os
import pandas as pd
from pathlib import Path


def test_data_visu_exists_after_run_script():
    train_visu = Path("data/processed/train_to_visu.csv")
    train_visu.resolve()
    subprocess.run(args=["make", "data_visu"])
    assert train_visu.is_file()


def test_data_transformed_not_lose_rows():
    train_visu = Path("data/processed/train_to_visu.csv")
    train_visu.resolve()
    train = Path("data/raw/train.csv")
    train.resolve()
    train_visu_data = pd.read_csv(filepath_or_buffer=train_visu)
    train_data = pd.read_csv(filepath_or_buffer=train)
    assert train_visu_data.shape[0] == train_data.shape[0]


def test_can_access_to_fields_with_long_name():
    train_visu = Path("data/processed/train_to_visu.csv")
    train_visu.resolve()
    train_visu_data = pd.read_csv(filepath_or_buffer=train_visu)
    field_with_long_name = train_visu_data["#Hermanos o conyuges a bordo"]
    assert field_with_long_name.name == "#Hermanos o conyuges a bordo"


def test_data_transformed_have_eleven_columns():
    train_visu = Path("data/processed/train_to_visu.csv")
    train_visu.resolve()
    train_visu_data = pd.read_csv(filepath_or_buffer=train_visu)
    assert train_visu_data.shape[1] == 11
