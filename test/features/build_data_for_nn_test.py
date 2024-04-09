import pathlib as pl
from src.features.build_data_for_nn import build_data_for_nn


def test_exist_dir_after_run_script():
    data_raw = pl.Path("data/raw/train.csv")
    folder_data_processed = pl.Path("data/processed")

    data_raw.resolve()
    folder_data_processed.resolve()

    build_data_for_nn(
        data_raw_path=data_raw,
        data_processed_folder=folder_data_processed,
    )

    assert folder_data_processed.is_dir()
