from pathlib import Path
import click
import kaggle


@click.command
@click.argument("data_raw_folder_path", type=click.Path(path_type=Path))
def main(data_raw_folder_path: Path):
    data_raw_folder_path.resolve()
    make_dataset(data_raw_folder_path)


def make_dataset(data_raw_folder_path: Path):
    kaggle.api.competition_download_file(
        competition="titanic", file_name="train.csv", path=data_raw_folder_path
    )
    kaggle.api.competition_download_file(
        competition="titanic", file_name="test.csv", path=data_raw_folder_path
    )
    kaggle.api.competition_download_file(
        competition="titanic",
        file_name="gender_submission.csv",
        path=data_raw_folder_path,
    )


if __name__ == "__main__":
    main()
