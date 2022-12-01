import json
import re
from statistics import mean
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel

SC = load("scaler.pickle")
RIGDE = load("model.pickle")
OHE = load("encoder.pickle")
COLS_OHE = list(OHE.get_feature_names_out())
COLS_CAT = ["fuel", "seller_type", "transmission", "m_units", "t_units", "seats"]
COLS_ALL = [
    "name",
    "year",
    "selling_price",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage",
    "engine",
    "max_power",
    "torque",
    "seats",
]

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


def unpack_data(item):
    return (
        item.name,
        item.year,
        item.selling_price,
        item.km_driven,
        item.fuel,
        item.seller_type,
        item.transmission,
        item.owner,
        item.mileage,
        item.engine,
        item.max_power,
        item.torque,
        item.seats,
    )


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    data = pd.DataFrame(columns=COLS_ALL)
    data.loc[0, :] = unpack_data(item=item)
    data = prep_data(data=data)
    pred = make_pred(data=data)
    return [float(i) for i in list(eval(pred))][0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    data = pd.DataFrame(columns=COLS_ALL)
    for item in items:
        data.loc[len(data), :] = unpack_data(item=item)
    data = prep_data(data=data)
    pred = make_pred(data=data)
    return [float(i) for i in list(eval(pred))]


def nm_torque(x):
    if "nm" in x:
        return "nm"
    elif "nn" in x:
        return "nn"
    elif "kgm" in x:
        return "kgm"
    else:
        return np.nan


def rpm_torque(x):
    if "rpm" in x:
        return "rpm"
    else:
        return np.nan


def prep_data(data: pd.DataFrame):
    data[["mileage", "m_units"]] = data["mileage"].str.split(" ", n=1, expand=True)
    data[["engine", "e_units"]] = data["engine"].str.split(" ", n=1, expand=True)
    data[["max_power", "p_units"]] = data["max_power"].str.split(" ", n=1, expand=True)
    data.loc[data[data["max_power"] == ""].index, "max_power"] = np.nan
    data["torque_vals"] = data.apply(
        lambda x: [i for i in re.sub("[^0-9.]", " ",
        re.sub(",", ".", str(x["torque"]))).split(" ") if i != ""],
        axis=1,
    )
    data["torque1"] = data.apply(
        lambda x: float(x["torque_vals"][0]) if len(x["torque_vals"]) > 0 else np.nan,
        axis=1,
    )
    data["torque2"] = data.apply(
        lambda x: mean([float(i) for i in x["torque_vals"][1:]])
        if len(x["torque_vals"][1:]) > 0
        else np.nan,
        axis=1,
    )
    data["t_units"] = data.apply(
        lambda x: nm_torque(
            (re.sub("[^nrpmkg ]", "", str(x["torque"]).lower())).split()),
        axis=1,
    )
    data[["mileage", "engine", "max_power"]] = data[
        ["mileage", "engine", "max_power"]
    ].astype("float")
    data = data.drop(columns=["torque", "torque_vals", "p_units", "e_units"])
    return data


def make_pred(data: pd.DataFrame):
    data = data.drop(columns=["selling_price", "name", "owner"])
    data = pd.concat(
        (
            data.drop(columns=COLS_CAT),
            pd.DataFrame(data=OHE.transform(data[COLS_CAT]), columns=COLS_OHE),
        ),
        axis=1,
    )
    data = SC.transform(data)
    pred = RIGDE.predict(data)
    return json.dumps(list(pred))
