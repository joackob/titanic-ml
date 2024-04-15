import pathlib as pl
import pandas as pd
from src.features.build_data_for_nn import build_data_for_nn

data_raw_path = pl.Path("data/raw/train.csv")
data_raw_path.resolve()
data_raw = pd.read_csv(filepath_or_buffer=data_raw_path)

(
    feature_space,
    dataset_train,
    dataset_validation,
) = build_data_for_nn(titanic_data=data_raw)


## unit test
def test_build_data_train_in_batchs():
    batch_size = 32
    for data, _ in dataset_train.take(count=1):
        assert len(data) == batch_size


def test_build_data_validation_in_batchs():
    batch_size = 32
    for data, _ in dataset_validation.take(count=1):
        assert len(data) == batch_size
