import keras as kr
import pandas as pd
import tensorflow as tf
import pathlib as pl
import click


@click.command
@click.argument("data_raw_path", type=click.Path(path_type=pl.Path, exists=True))
@click.argument("data_processed_folder", type=click.Path(path_type=pl.Path))
def main(data_raw_path: pl.Path, data_processed_folder: pl.Path):
    data_raw_path.resolve()
    data_processed_folder.resolve()
    build_data_for_nn(data_raw_path, data_processed_folder)


def build_data_for_nn(data_raw_path: pl.Path, data_processed_folder: pl.Path):
    data_raw = pd.read_csv(filepath_or_buffer=data_raw_path)
    features = data_raw.copy().loc[
        :,
        [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
        ],
    ]

    features.loc[features["Embarked"].isna(), "Embarked"] = "D"
    labels = data_raw.copy().pop(item="Survived")

    features = features.astype(dtype={"Sex": "string", "Embarked": "string"})

    inputs = {
        "Pclass": kr.Input(
            shape=(1,),
            dtype=features.dtypes["Pclass"],
            name="Pclass",
        ),
        "Sex": kr.Input(
            shape=(1,),
            dtype=features.dtypes["Sex"],
            name="Sex",
        ),
        "Age": kr.Input(
            shape=(1,),
            dtype=features.dtypes["Age"],
            name="Age",
        ),
        "SibSp": kr.Input(
            shape=(1,),
            dtype=features.dtypes["SibSp"],
            name="SibSp",
        ),
        "Parch": kr.Input(
            shape=(1,),
            dtype=features.dtypes["Parch"],
            name="Parch",
        ),
        "Fare": kr.Input(
            shape=(1,),
            dtype=features.dtypes["Fare"],
            name="Fare",
        ),
        "Embarked": kr.Input(
            shape=(1,),
            dtype=features.dtypes["Embarked"],
            name="Embarked",
        ),
    }

    pclass_output = kr.layers.CategoryEncoding(num_tokens=3)
    gender_output = kr.layers.StringLookup(vocabulary=features["Sex"].unique())
    age_output = kr.layers.Discretization(bin_boundaries=[0, 6, 12, 18, 26, 59])
    sibsp_output = kr.layers.Normalization()
    parch_output = kr.layers.Normalization()
    fare_output = kr.layers.Normalization()
    embarked_output = kr.layers.StringLookup(vocabulary=features["Embarked"].unique())

    outputs = {
        "Pclass": pclass_output(inputs["Pclass"]),
        "Sex": gender_output(inputs["Sex"]),
        "Age": age_output(inputs["Age"]),
        "SibSp": sibsp_output(inputs["SibSp"]),
        "Parch": parch_output(inputs["Parch"]),
        "Fare": fare_output(inputs["Fare"]),
        "Embarked": embarked_output(inputs["Embarked"]),
    }

    preprocessing_model = kr.Model(inputs, outputs)
    features_processed = preprocessing_model(dict(features))
    dataset = tf.data.Dataset.from_tensor_slices(
        tensors=(dict(features_processed), labels)
    )
    dataset.save(path=str(data_processed_folder))


if __name__ == "__main__":
    main()
