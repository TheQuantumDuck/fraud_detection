import pickle
from typing import Any, List

import numpy as np

class Model:
    
    def __init__(self, model: Any):
        self.model = model
        self.use_cols = []

    def predict(
        self,
        country_of_origin: int,
        is_nigeria: int,
        is_postal_100001: int,
        is_duplicate: int,
        is_after_hours: int,
        category: str
    ) -> float:
        raise NotImplementedError()

    def save(self, name: str) -> None:
        with open(f"./models/{name}.pickle", "wb") as file:
            pickle.dump(self, file) 

    @staticmethod
    def load(name: str) -> "Model":
        with open(f"./models/{name}.pickle", "rb") as file:
            model = pickle.load(file)
        return model


class LogReg(Model):

    def predict(
        self,
        country_of_origin: int,
        is_nigeria: int,
        is_postal_100001: int,
        is_duplicate: int,
        is_after_hours: int,
        category: str
    ) -> float:
        x = np.array([[
            is_nigeria,
            is_duplicate,
            0,
            0,
            0
        ]], dtype=int)
        match category.lower():
            case "electronic":
                x[0, 2] = 1
            case "houseware":
                x[0, 3] = 1
            case "services":
                x[0, 4] = 1
            case _:
                pass
        return self.model.predict(x).item()


class TreeMod(Model):

    def predict(
        self,
        country_of_origin: int,
        is_nigeria: int,
        is_postal_100001: int,
        is_duplicate: int,
        is_after_hours: int,
        category: str
    ) -> float:
        x = np.array([[
            country_of_origin,
            is_nigeria,
            is_postal_100001,
            is_duplicate,
            is_after_hours,
            0,
            0,
            0,
            0
        ]], dtype=int)
        match category.lower():
            case "electronic":
                x[0, 5] = 1
            case "houseware":
                x[0, 6] = 1
            case "services":
                x[0, 8] = 1
            case _:
                x[0, 7] = 1
        return self.model.predict_proba(x)[0, 1]