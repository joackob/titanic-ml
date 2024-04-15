import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import pandas as pd
import keras as kr
import pathlib as pl
import click


@click.command
@click.argument("data_raw_path", type=click.Path(path_type=pl.Path, exists=True))
@click.argument("data_processed_folder", type=click.Path(path_type=pl.Path))
def main(data_raw_path: pl.Path, data_processed_folder: pl.Path):
    data_raw_path.resolve()
    titanic_data = pd.read_csv(filepath_or_buffer=data_raw_path)

    (
        feature_space,
        dataset_train,
        dataset_validation,
    ) = build_data_for_nn(titanic_data)

    data_processed_folder.resolve()
    if not data_processed_folder.is_dir():
        data_processed_folder.mkdir()

    feature_space_path = data_processed_folder / "feature_space.keras"
    dataset_train_path = data_processed_folder / "train.keras"
    dataset_validation_path = data_processed_folder / "val.keras"

    feature_space_path.resolve()
    dataset_train_path.resolve()
    dataset_validation_path.resolve()

    feature_space.save(
        filepath=feature_space_path.__str__(),
    )
    dataset_train.save(
        path=dataset_train_path.__str__(),
    )
    dataset_validation.save(
        path=dataset_validation_path.__str__(),
    )


def build_data_for_nn(titanic_data: pd.DataFrame):
    data = titanic_data.copy()
    data.loc[data["Embarked"].isna(), "Embarked"] = "D"
    data = data.loc[
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

    titanic_data_validation = data.sample(frac=0.2)
    titanic_data_train = data.drop(titanic_data_validation.index)

    titanic_dataset_train = dataframe_to_dataset(
        dataframe=titanic_data_train, target_col="Survived"
    )
    titanic_dataset_validation = dataframe_to_dataset(
        dataframe=titanic_data_validation, target_col="Survived"
    )

    titanic_dataset_train = titanic_dataset_train.batch(batch_size=32)
    titanic_dataset_validation = titanic_dataset_validation.batch(batch_size=32)

    feature_space = kr.utils.FeatureSpace(
        features={
            "Pclass": kr.utils.FeatureSpace.integer_categorical(num_oov_indices=0),
            "Sex": kr.utils.FeatureSpace.string_categorical(num_oov_indices=0),
            "Age": kr.utils.FeatureSpace.float_discretized(num_bins=8),
            "SibSp": kr.utils.FeatureSpace.float_normalized(),
            "Parch": kr.utils.FeatureSpace.float_normalized(),
            "Fare": kr.utils.FeatureSpace.float_normalized(),
            "Embarked": kr.utils.FeatureSpace.string_categorical(num_oov_indices=0),
        },
        crosses=[
            kr.utils.FeatureSpace.cross(
                feature_names=(
                    "Sex",
                    "Age",
                ),
                crossing_dim=16,
            ),
        ],
        output_mode="concat",
    )

    titanic_dataset_train_without_target = titanic_dataset_train.map(
        map_func=lambda data, _: data
    )
    feature_space.adapt(
        dataset=titanic_dataset_train_without_target,
    )
    preprocessed_dataset_train = titanic_dataset_train.map(
        map_func=lambda data, target: (feature_space(data), target)
    )
    preprocessed_dataset_validation = titanic_dataset_validation.map(
        map_func=lambda data, target: (feature_space(data), target)
    )

    return (
        feature_space,
        preprocessed_dataset_train,
        preprocessed_dataset_validation,
    )


def dataframe_to_dataset(
    dataframe: pd.DataFrame | pd.Series,
    target_col: str,
) -> tf.data.Dataset:
    dataframe = dataframe.copy()
    labels = dataframe.pop(item=target_col)
    ds = tf.data.Dataset.from_tensor_slices(tensors=(dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=dataframe.shape[0])
    return ds


if __name__ == "__main__":
    main()
