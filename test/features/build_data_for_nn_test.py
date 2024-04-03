import subprocess
import pandas as pd
import numpy as np
from pathlib import Path


path_dst_file = "data/interim/train_for_nn.csv"
data_train_path = Path(path_dst_file)
data_train_path.resolve()


def test_data_train_nn_exist_after_run_make():
    subprocess.run(args=["make", "data_train_nn"])
    assert data_train_path.is_file()


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
