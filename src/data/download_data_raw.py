from pathlib import Path
import click
import kaggle
import logging


@click.command
@click.argument("folder_data_raw", type=click.Path(path_type=Path))
def main(folder_data_raw: Path):
    folder_data_raw.resolve()
    kaggle.api.competition_download_file(
        competition="titanic", file_name="train.csv", path=folder_data_raw
    )
    kaggle.api.competition_download_file(
        competition="titanic", file_name="test.csv", path=folder_data_raw
    )
    kaggle.api.competition_download_file(
        competition="titanic", file_name="gender_submission.csv", path=folder_data_raw
    )


if __name__ == "__main__":
    main()
