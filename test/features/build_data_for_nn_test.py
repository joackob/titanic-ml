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
def test_build_data_with_same_dims_data_raw():
    assert len(dataset_train) + len(dataset_validation) == len(data_raw)
