import click
import kaggle
import pathlib as pl


@click.command
@click.argument("predictions_path", type=click.Path(path_type=pl.Path))
def main(predictions_path: pl.Path):
    predictions_path.resolve()
    make_submission(predictions_path)


def make_submission(predictions_path: pl.Path):
    kaggle.api.competition_submit(
        file_name=predictions_path,
        competition="titanic",
        message="algo",
    )


if __name__ == "__main__":
    main()
