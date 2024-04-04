import click
import pandas as pd
import numpy as np
from pathlib import Path


@click.command
@click.argument("data_raw_path", type=click.Path(path_type=Path, exists=True))
@click.argument("folder_interim_path", type=click.Path(path_type=Path))
def main(data_raw_path: Path, folder_interim_path: Path):
    # transform data raw in data train and label train
    data_raw_path.resolve()
    data_raw = pd.read_csv(filepath_or_buffer=data_raw_path)
    data_procesed = (
        data_raw.copy()
        .loc[
            :,
            [
                "PassengerId",
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
        .replace(
            to_replace={
                "Sex": {"male": 0, "female": 1},
                "Embarked": {
                    "C": 0,
                    "Q": 1,
                    "S": 2,
                },
            }
        )
    )

    # categorizo la edad en rangos
    data_procesed["Age"] = data_procesed["Age"].apply(age_by_range)

    # normalizar la tarifa
    mean_fare = data_procesed["Fare"].mean()
    std_fare = data_procesed["Fare"].std()
    data_procesed["Fare"] = (data_procesed["Fare"] - mean_fare) / std_fare

    # normalización rapida del resto de los campos
    data_procesed["Pclass"] = data_procesed["Pclass"] / data_procesed["Pclass"].max()
    data_procesed["Age"] = data_procesed["Age"] / data_procesed["Age"].max()
    data_procesed["SibSp"] = data_procesed["SibSp"] / data_procesed["SibSp"].max()
    data_procesed["Parch"] = data_procesed["Parch"] / data_procesed["Parch"].max()
    data_procesed["Embarked"] = (
        data_procesed["Embarked"] / data_procesed["Embarked"].max()
    )

    # balanceamos la cantidad de fallecidos con la cantidad de sobrevivientes
    data_survived = data_procesed.query("Survived == 1")
    data_no_survived = data_procesed.query("Survived == 0")
    data_train = pd.concat(
        objs=[data_survived, data_no_survived.sample(n=len(data_survived))]
    )
    data_test = pd.concat(
        objs=[data_survived, data_no_survived.sample(n=len(data_survived))]
    )

    # guardo los archivos correspondientes a los datos y los labels de entrenamiento
    data_train_path = folder_interim_path / "train_data_nn.csv"
    labels_train_path = folder_interim_path / "train_labels_nn.csv"
    data_test_path = folder_interim_path / "test_data_nn.csv"
    labels_test_path = folder_interim_path / "test_labels_nn.csv"
    data_train_path.resolve()
    labels_train_path.resolve()
    data_test_path.resolve()
    labels_test_path.resolve()
    folder_interim_path.resolve()

    if not folder_interim_path.is_dir():
        folder_interim_path.mkdir()

    data_train.loc[
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
    ].to_csv(path_or_buf=data_train_path, index=False)
    data_train.loc[
        :,
        [
            "PassengerId",
            "Survived",
        ],
    ].to_csv(path_or_buf=labels_train_path, index=False)

    data_test.loc[
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
    ].to_csv(path_or_buf=data_test_path, index=False)
    data_test.loc[
        :,
        [
            "PassengerId",
            "Survived",
        ],
    ].to_csv(path_or_buf=labels_test_path, index=False)


def age_by_range(age: float) -> int:
    if np.isnan(age):
        return -1
    elif age >= 0 and age < 6:
        return 0
    elif age >= 6 and age < 12:
        return 1
    elif age >= 12 and age < 18:
        return 2
    elif age >= 18 and age < 26:
        return 3
    elif age > 26 and age < 59:
        return 4
    else:
        return 5


if __name__ == "__main__":
    main()
