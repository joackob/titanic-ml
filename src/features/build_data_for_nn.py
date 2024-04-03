import click
import pandas as pd
import numpy as np
from pathlib import Path


@click.command
@click.argument("data_raw_path", type=click.Path(path_type=Path, exists=True))
@click.argument("data_transformed_path", type=click.Path(path_type=Path))
def main(data_raw_path: Path, data_transformed_path: Path):
    data_raw_path.resolve()
    data_transformed_path.resolve()
    data_train = pd.read_csv(filepath_or_buffer=data_raw_path)
    data_train_nn = (
        data_train.copy()
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
    data_train_nn["Age"] = data_train_nn["Age"].apply(age_by_range)

    # normalizar la tarifa
    mean_fare = data_train_nn["Fare"].mean()
    std_fare = data_train_nn["Fare"].std()

    data_train_nn["Fare"] = (data_train_nn["Fare"] - mean_fare) / std_fare

    # balanceamos la cantidad de fallecidos con la cantidad de sobrevivientes
    data_train_suvived = data_train_nn.query("Survived == 1")
    data_train_no_survived = data_train_nn.query("Survived == 0").head(
        n=len(data_train_suvived)
    )
    data_train_nn = pd.concat(objs=[data_train_suvived, data_train_no_survived])

    if not data_transformed_path.parent.is_dir():
        data_transformed_path.parent.mkdir()
    data_train_nn.to_csv(path_or_buf=data_transformed_path, index=False)


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
