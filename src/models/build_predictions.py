import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import keras as kr
import tensorflow as tf
import pandas as pd
import pathlib as pl
import numpy as np
import click


@click.command
@click.argument(
    "titanic_data_test_path",
    type=click.Path(path_type=pl.Path, exists=True),
)
@click.argument(
    "model_inference_path",
    type=click.Path(path_type=pl.Path, exists=True),
)
@click.argument(
    "predictions_folder",
    type=click.Path(path_type=pl.Path),
)
def main(
    titanic_data_test_path: pl.Path,
    inference_model_path: pl.Path,
    predictions_folder: pl.Path,
):
    titanic_data_test_path.resolve()
    inference_model_path.resolve()

    titanic_data_test = pd.read_csv(filepath_or_buffer=titanic_data_test_path)
    inference_model = kr.models.load_model(filepath=inference_model_path)

    predictions = build_predictions(
        data_test=titanic_data_test,
        inference_model=inference_model,
    )
    if not predictions_folder.is_dir():
        predictions_folder.mkdir()

    predictions_path = predictions_folder / "predictions.csv"
    predictions_path.resolve()
    predictions.to_csv(path_or_buf=predictions_path, index=False)


def build_predictions(
    data_test: pd.DataFrame,
    inference_model: kr.models.Model,
) -> pd.DataFrame:
    data = data_test.copy()
    data.loc[data["Embarked"].isna(), "Embarked"] = "D"
    data = data.loc[
        :,
        [
            "PassengerId",
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
        ],
    ]
    ids = data.drop(labels="PassengerId")
    dataset = tf.data.Dataset.from_tensors(tensors=dict(data))
    predictions = inference_model.predict(x=dataset)
    predictions = np.reshape(a=predictions, newshape=len(predictions))
    predictions = np.where(predictions > 0.5, 1, 0)
    return pd.DataFrame(data={"PassengerId": ids, "Survived": predictions})
