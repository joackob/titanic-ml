import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import keras as kr

# import tensorflow.python.keras as kr
import numpy as np
import pandas as pd
import click
from pathlib import Path


@click.command
@click.argument("data_raw_path", type=click.Path(path_type=Path, exists=True))
@click.argument("folder_models_path", type=click.Path(path_type=Path))
def main(data_raw_path: Path, folder_models_path: Path):
    # transform data raw in data train and label train
    data_raw_path.resolve()
    data_raw = pd.read_csv(filepath_or_buffer=data_raw_path)
    data_procesed = data_raw.copy().loc[
        :,
        [
            "Survived",
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
        ],
    ]

    data_target = data_procesed.pop(item="Survived")

    inputs, outputs = build_pre_processing_model()
    pre_processing_model = kr.Model(inputs=inputs, outputs=outputs)
    fs_dense_layer = kr.layers.Dense(units=8, activation=kr.activations.relu)(
        pre_processing_model
    )
    sd_dense_layer = kr.layers.Dense(units=8, activation=kr.activations.relu)(
        fs_dense_layer
    )
    th_dense_layer = kr.layers.Dense(units=2, activation=kr.activations.softmax)(
        sd_dense_layer
    )

    model = kr.Model(inputs, th_dense_layer)

    model.summary()


def build_pre_processing_model():
    # https://www.tensorflow.org/tutorials/structured_data/feature_columns?hl=es-419
    # https://www.tensorflow.org/guide/migrate/migrating_feature_columns
    # En recientes versiones, Keras propone que el preprocesamiento de los datos
    # sea mediante capas, creando un modelo en si mismo
    inputs = {
        "Pclass": kr.Input(shape=(), dtype="int32"),
        "Sex": kr.Input(shape=(), dtype="string"),
        "Age": kr.Input(shape=(), dtype="float32"),
        "SibSp": kr.Input(shape=(2, 1), dtype="int32"),
        "Parch": kr.Input(shape=(2, 1), dtype="int32"),
        "Fare": kr.Input(shape=(), dtype="float32"),
        "Embarked": kr.Input(shape=(), dtype="int32"),
    }

    pclass_output = kr.layers.CategoryEncoding(num_tokens=3)
    gender_output = kr.layers.StringLookup(max_tokens=2, output_mode="one_hot")
    age_output = kr.layers.Discretization(bin_boundaries=[0, 6, 12, 18, 26, 59])
    sibsp_output = kr.layers.Normalization(mean=0, variance=1)
    parch_output = kr.layers.Normalization(mean=0, variance=1)
    fare_output = kr.layers.Normalization(mean=0, variance=1)
    embarked_output = kr.layers.CategoryEncoding(num_tokens=3)

    outputs = {
        "Pclass": pclass_output(inputs["Pclass"]),
        "Sex": gender_output(inputs["Sex"]),
        "Age": age_output(inputs["Age"]),
        "SibSp": sibsp_output(inputs["SibSp"]),
        "Parch": parch_output(inputs["Parch"]),
        "Fare": fare_output(inputs["Fare"]),
        "Embarked": embarked_output(inputs["Embarked"]),
    }

    return (inputs, outputs)


if __name__ == "__main__":
    main()
