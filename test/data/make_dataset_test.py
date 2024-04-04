from pathlib import Path
import subprocess as sp


def test_kaggle_json_exist():
    home = Path.home()
    kaggle_cred = home / ".kaggle" / "kaggle.json"
    kaggle_cred.resolve()
    assert kaggle_cred.is_file()


def test_data_exists_after_run_script_for_download():
    train = Path("data/raw/train.csv")
    train.resolve()
    sp.run(args=["make", "data"])
    assert train.is_file()
