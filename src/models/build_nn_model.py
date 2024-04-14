import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import keras as kr
import tensorflow as tf
import pandas as pd
import pathlib as pl
import click


@click.command
@click.argument("dataset_folder", type=click.Path(path_type=pl.Path, exists=True))
@click.argument("models_folder", type=click.Path(path_type=pl.Path))
def main(dataset_folder: pl.Path, models_folder: pl.Path):
    dataset_folder.resolve()
    feature_space_path = dataset_folder / "feature_space.keras"
    dataset_train_path = dataset_folder / "train.keras"
    dataset_validation_path = dataset_folder / "val.keras"
    feature_space_path.resolve()
    dataset_train_path.resolve()
    dataset_validation_path.resolve()
    feature_space = kr.models.load_model(filepath=feature_space_path)
    titanic_dataset_train = tf.data.Dataset.load(
        path=dataset_train_path.__str__(),
    )
    titanic_dataset_validation = tf.data.Dataset.load(
        path=dataset_validation_path.__str__()
    )
    training_model, inference_model = build_nn_model(
        feature_space,
        titanic_dataset_train,
        titanic_dataset_validation,
    )

    models_folder.resolve()
    if not models_folder.is_dir():
        models_folder.mkdir()

    training_model_path = models_folder / "training_model.keras"
    inference_model_path = models_folder / "inference_model.keras"
    training_model_path.resolve()
    inference_model_path.resolve()

    training_model.save(filepath=training_model_path)
    inference_model.save(filepath=inference_model_path)


def build_nn_model(
    feature_space: kr.utils.FeatureSpace,
    titanic_dataset_train: tf.data.Dataset,
    titanic_dataset_validation: tf.data.Dataset,
):
    dict_inputs = feature_space.get_inputs()
    encoded_features = feature_space.get_encoded_features()

    layers = kr.models.Sequential(
        layers=[
            kr.layers.Dense(units=32, activation=kr.activations.relu),
            kr.layers.Dense(units=32, activation=kr.activations.relu),
            kr.layers.Dense(units=32, activation=kr.activations.relu),
            kr.layers.Dense(units=32, activation=kr.activations.relu),
            kr.layers.Dense(units=32, activation=kr.activations.relu),
            kr.layers.Dropout(rate=0.5),
            kr.layers.Dense(units=1, activation=kr.activations.sigmoid),
        ]
    )

    predictions = layers(encoded_features)

    training_model = kr.Model(
        inputs=encoded_features,
        outputs=predictions,
    )
    training_model.compile(
        optimizer=kr.optimizers.Adam(),
        loss=kr.losses.binary_crossentropy,
        metrics=[kr.metrics.BinaryAccuracy()],
    )
    inference_model = kr.Model(
        inputs=dict_inputs,
        outputs=predictions,
    )
    training_model.fit(
        titanic_dataset_train,
        epochs=20,
        validation_data=titanic_dataset_validation,
        verbose=2,
    )

    return (
        training_model,
        inference_model,
    )


if __name__ == "__main__":
    main()
