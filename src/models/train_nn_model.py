import tensorflow as tf
import keras as kr
import numpy as np
import pandas as pd
import click

from pathlib import Path


@click.command
@click.argument("folder_interim_path", type=click.Path(path_type=Path, exists=True))
@click.argument("folder_models_path", type=click.Path(path_type=Path))
def main(folder_interim_path: Path, folder_models_path: Path):

    (data_train, labels_train, data_test, labels_test) = load_data(folder_interim_path)

    NN_INTERIM_LAYERS = 16
    N_ANSWERS = 2
    model = kr.models.Sequential(
        layers=[
            kr.layers.Input(shape=data_train.shape),
            kr.layers.Dense(units=NN_INTERIM_LAYERS, activation=kr.activations.relu),
            kr.layers.Dense(units=NN_INTERIM_LAYERS, activation=kr.activations.relu),
            kr.layers.Dense(units=NN_INTERIM_LAYERS, activation=kr.activations.relu),
            kr.layers.Dense(units=NN_INTERIM_LAYERS, activation=kr.activations.relu),
            kr.layers.Dropout(rate=0.2),
            kr.layers.Dense(units=N_ANSWERS, activation=kr.activations.softmax),
        ]
    )

    model.compile(
        loss=kr.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=kr.optimizers.RMSprop(),
        metrics=[kr.metrics.Accuracy()],
    )

    model.fit(data_train, labels_train, batch_size=64, epochs=2, validation_split=0.2)
    model.evaluate(data_test, labels_test, verbose=2)
    # save model
    folder_models_path.resolve()
    if not folder_models_path.is_dir():
        folder_models_path.mkdir()

    model_nn = folder_models_path / "model_nn.keras"
    model_nn.resolve()

    model.save(filepath=model_nn)


def load_data(folder_interim_path: Path):
    data_train_path = folder_interim_path / "train_data_nn.csv"
    labels_train_path = folder_interim_path / "train_labels_nn.csv"
    data_test_path = folder_interim_path / "test_data_nn.csv"
    labels_test_path = folder_interim_path / "test_labels_nn.csv"

    data_train_path.resolve()
    labels_train_path.resolve()
    data_test_path.resolve()
    labels_test_path.resolve()

    data_train = pd.read_csv(data_train_path)
    labels_train = pd.read_csv(labels_train_path)
    data_test = pd.read_csv(data_test_path)
    labels_test = pd.read_csv(labels_test_path)

    data_train = data_train.loc[:, data_train.columns != "PassengerId"].to_numpy(
        dtype=np.float16
    )
    data_test = data_test.loc[:, data_test.columns != "PassengerId"].to_numpy(
        dtype=np.float16
    )
    labels_train = labels_train["Survived"].to_numpy(dtype=np.float16)
    labels_test = labels_test["Survived"].to_numpy(dtype=np.float16)

    return (data_train, labels_train, data_test, labels_test)
