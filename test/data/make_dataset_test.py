import pathlib as pl
from src.data.make_dataset import make_dataset


def test_kaggle_json_exist():
    home = pl.Path.home()
    kaggle_cred = home / ".kaggle" / "kaggle.json"
    kaggle_cred.resolve()
    assert kaggle_cred.is_file()


def test_data_train_exists_after_run_script_for_download():
    data_raw_folder_path = pl.Path("data/raw")
    data_raw_folder_path.resolve()

    make_dataset(data_raw_folder_path)

    data_raw_path = pl.Path("data/raw/train.csv")
    data_raw_path.resolve()
    assert data_raw_path.is_file()


def test_data_test_exists_after_run_script_for_download():
    data_raw_folder_path = pl.Path("data/raw")
    data_raw_folder_path.resolve()

    make_dataset(data_raw_folder_path)

    data_raw_path = pl.Path("data/raw/test.csv")
    data_raw_path.resolve()
    assert data_raw_path.is_file()
