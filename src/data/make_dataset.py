from pathlib import Path
import click
import kaggle


@click.command
@click.argument("folder_data_raw_path", type=click.Path(path_type=Path))
def main(folder_data_raw_path: Path):
    folder_data_raw_path.resolve()
    kaggle.api.competition_download_file(
        competition="titanic", file_name="train.csv", path=folder_data_raw_path
    )
    kaggle.api.competition_download_file(
        competition="titanic", file_name="test.csv", path=folder_data_raw_path
    )
    kaggle.api.competition_download_file(
        competition="titanic",
        file_name="gender_submission.csv",
        path=folder_data_raw_path,
    )


if __name__ == "__main__":
    main()
