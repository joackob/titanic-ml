import click
import kaggle
import pathlib as pl


@click.command
@click.argument("predictions_path", type=click.Path(path_type=pl.Path))
@click.argument("metadata", type=click.STRING)
def main(predictions_path: pl.Path, metadata: str):
    predictions_path.resolve()
    make_submission(predictions_path, metadata)


def make_submission(predictions_path: pl.Path, metadata: str):
    kaggle.api.competition_submit(
        file_name=predictions_path,
        competition="titanic",
        message=metadata,
    )


if __name__ == "__main__":
    main()
