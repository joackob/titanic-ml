from src.models.build_inference_model import build_inference_model

import pandas as pd
import pathlib as pl
import keras as kr
import tensorflow as tf
import numpy as np

feature_space_path = pl.Path("data/processed/feature_space.keras")
train_dataset_path = pl.Path("data/processed/train_dataset.keras")
validation_dataset_path = pl.Path("data/processed/validation_dataset.keras")
data_raw_path = pl.Path("data/raw/train.csv")

feature_space_path.resolve()
train_dataset_path.resolve()
validation_dataset_path.resolve()
data_raw_path.resolve()

feature_space = kr.models.load_model(filepath=feature_space_path)
train_dataset = tf.data.Dataset.load(path=train_dataset_path.__str__())
validation_dataset = tf.data.Dataset.load(path=validation_dataset_path.__str__())
data_raw = pd.read_csv(filepath_or_buffer=data_raw_path)

inference_model = build_inference_model(
    feature_space=feature_space,
    titanic_dataset_train=train_dataset,
    titanic_dataset_validation=validation_dataset,
)

sample = data_raw.sample(n=1)
sample = sample.loc[
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
target = sample.pop(item="Survived")
input = {name: tf.convert_to_tensor(value) for name, value in sample.items()}


def test_inference_model_can_predict_one_case():
    prediction = inference_model.predict(input)
    prediction = np.where(prediction > 0.5, 1, 0)
    assert prediction[0][0] == target.iloc[0]
