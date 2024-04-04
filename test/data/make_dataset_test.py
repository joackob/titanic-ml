from pathlib import Path
import subprocess as sp


def test_data_exists_after_run_script_for_download():
    train = Path("data/raw/train.csv")
    train.resolve()
    sp.run(args=["make", "data"])
    assert train.is_file()
