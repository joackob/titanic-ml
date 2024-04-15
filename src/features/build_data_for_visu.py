import click
import pandas as pd
import numpy as np
import pathlib as pl


@click.command
@click.argument("data_raw_path", type=click.Path(path_type=pl.Path, exists=True))
@click.argument("data_processed_folder", type=click.Path(path_type=pl.Path))
def main(data_raw_path: pl.Path, data_processed_folder: pl.Path):
    data_raw_path.resolve()
    data_raw = pd.read_csv(filepath_or_buffer=data_raw_path)

    data_processed = build_data_for_visu(data_raw)

    data_processed_folder.resolve()
    if not data_processed_folder.is_dir():
        data_processed_folder.mkdir()

    data_processed_path = data_processed_folder / "data_for_visu.csv"
    data_processed_path.resolve()
    data_processed.to_csv(path_or_buf=data_processed_path, index=False)


def build_data_for_visu(data_raw: pd.DataFrame) -> pd.DataFrame:
    data = data_raw.copy()
    data = (
        data.loc[
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
    data["Rango etario"] = data["Edad"].apply(rango_etario)
    return data


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
