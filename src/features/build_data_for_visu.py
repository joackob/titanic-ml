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
    data_train_to_visu = (
        data_train.loc[
            :,
            [
                "PassengerId",
                "Survived",
                "Pclass",
                "Sex",
                "Age",
                "SibSp",
                "Parch",
                "Ticket",
                "Fare",
                "Cabin",
                "Embarked",
            ],
        ]
        .rename(
            columns={
                "PassengerId": "IdPasajero",
                "Survived": "Condicion",
                "Pclass": "Clase",
                "Sex": "Genero",
                "Age": "Edad",
                "SibSp": "#Hermanos o conyuges a bordo",
                "Parch": "#Padres o hijos a bordo",
                "Ticket": "Ticket",
                "Fare": "Tarifa",
                "Cabin": "Cabina",
                "Embarked": "Puerto de embarcacion",
            }
        )
        .replace(
            to_replace={
                "Condicion": {0: "Fallecido", 1: "Sobreviviente"},
                "Clase": {1: "Primera", 2: "Segunda", 3: "Tercera"},
                "Genero": {"male": "Hombre", "female": "Mujer"},
                "Puerto de embarcacion": {
                    "C": "Cherbourg",
                    "Q": "Queenstown",
                    "S": "Southampton",
                },
            }
        )
    )
    data_train_to_visu["Rango etario"] = data_train_to_visu["Edad"].apply(rango_etario)

    if not data_transformed_path.parent.is_dir():
        data_transformed_path.parent.mkdir()
    data_train_to_visu.to_csv(path_or_buf=data_transformed_path, index=False)


def rango_etario(edad: float) -> str:
    def age_in_range(min: float, max: float) -> bool:
        return edad >= min and edad < max

    if np.isnan(edad):
        return "Desconocido"
    elif age_in_range(min=0, max=6):
        return "Primera infancia"
    elif age_in_range(min=6, max=12):
        return "Infancia"
    elif age_in_range(min=12, max=18):
        return "Adolecencia"
    elif age_in_range(min=18, max=26):
        return "Juventud"
    elif age_in_range(min=26, max=59):
        return "Adultez"
    else:
        return "Persona adulta"


if __name__ == "__main__":
    main()
