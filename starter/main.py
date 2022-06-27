import os
import joblib

from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd

from starter.ml.model import inference
from starter.ml.data import process_data

app = FastAPI()

model = joblib.load(os.path.join("model", "model.joblib"))
encoder = joblib.load(os.path.join("model", "encoder.joblib"))
lb = joblib.load(os.path.join("model", "lb.joblib"))


class Data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")
    salary: str

    class Config:
        allow_population_by_field_name = True


@app.get("/")
async def say_hello():
    return {"greeting": "Hello!"}


@app.post("/data/")
async def upload_data(data: Data):
    return data


@app.post("/predict/")
def get_prediction(payload: Data):

    pd_data = pd.DataFrame(payload.dict(), index=[0])
    pd_data.rename(
        columns={
            "age": "age",
            "workclass": "workclass",
            "fnlgt": "fnlgt",
            "education": "education",
            "education_num": "education-num",
            "marital_status": "marital-status",
            "occupation": "occupation",
            "relationship": "relationship",
            "race": "race",
            "sex": "sex",
            "capital_gain": "capital-gain",
            "capital_loss": "capital-loss",
            "hours_per_week": "hours-per-week",
            "native_country": "native-country",
            "salary": "salary",
        },
        inplace=True,
    )

    X, _, _, _ = process_data(pd_data, label="salary", encoder=encoder, lb=lb, training=False)

    prediction = inference(model=model, X=X)

    return {"prediction": prediction.tolist()}
