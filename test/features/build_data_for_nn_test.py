import subprocess
import pandas as pd
import numpy as np
from pathlib import Path


data_raw_path = Path("data/raw/train.csv")
folder_interim_path = Path("data/interim")
data_train_path = folder_interim_path / "data_train_nn.csv"
data_test_path = folder_interim_path / "data_test_nn.csv"
data_raw_path.resolve()
data_train_path.resolve()
data_test_path.resolve()
folder_interim_path.resolve()


def test_data_train_nn_exist_after_run_make():
    subprocess.run(args=["make", "data_nn"])
    assert data_train_path.is_file()


def test_data_test_nn_exist_after_run_make():
    subprocess.run(args=["make", "data_nn"])
    assert data_test_path.is_file()


def test_data_train_nn_have_same_number_of_s_ns():
    data_train = pd.read_csv(filepath_or_buffer=data_train_path)
    assert len(data_train.query("Survived == 0")) == len(
        data_train.query("Survived == 1")
    )


def test_data_train_fare_is_normalized():
    data_train = pd.read_csv(filepath_or_buffer=data_train_path)
    assert (
        np.round(data_train["Fare"].mean()) == 0
        and np.round(data_train["Fare"].std()) == 1
    )


def test_data_train_labels_is_same_at_data_raw():
    labels_train = pd.read_csv(filepath_or_buffer=data_train_path).rename(
        columns={"Survived": "Labels_Survived"}
    )
    data_raw = pd.read_csv(filepath_or_buffer=data_raw_path).rename(
        columns={"Survived": "Raw_Survived"}
    )
    sample = labels_train.sample(n=10)
    sample_to_test = pd.merge(sample, data_raw, on="PassengerId", how="left")
    result = sample_to_test["Labels_Survived"] == sample_to_test["Raw_Survived"]

    assert result.all()
