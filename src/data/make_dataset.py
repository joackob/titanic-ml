import pathlib as pl
import click
import kaggle


@click.command
@click.argument("data_raw_folder", type=click.Path(path_type=pl.Path))
def main(data_raw_folder: pl.Path):
    data_raw_folder.resolve()
    make_dataset(data_raw_folder)


def make_dataset(data_raw_folder: pl.Path):
    kaggle.api.competition_download_file(
        competition="titanic", file_name="train.csv", path=data_raw_folder
    )
    kaggle.api.competition_download_file(
        competition="titanic", file_name="test.csv", path=data_raw_folder
    )
    kaggle.api.competition_download_file(
        competition="titanic",
        file_name="gender_submission.csv",
        path=data_raw_folder,
    )


if __name__ == "__main__":
    main()
