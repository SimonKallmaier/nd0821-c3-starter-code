import os
import json
import joblib

from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd

from starter.ml.model import inference, compute_model_metrics
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


@app.get("/")
async def say_hello():
    return {"greeting": "Hello!"}


@app.post("/data/")
async def upload_data(data: Data):
    return data


@app.post("/predict", response_model=Data, status_code=200)
def get_prediction(payload: Data):
    

    pd_data = pd.DataFrame(payload.dict(), index=[0])
    

    X, y, _, _ = process_data(pd_data, encoder=encoder, lb=lb, training=False)
    
    print(X)
    print(y)

    prediction = inference(model=model, X=X)
    print(prediction)
    
    precision, recall, fbeta = compute_model_metrics(y=y, preds=prediction)

    response_object = {
        "prediction": prediction,
        "precision": precision,
        "recall": recall,
        "fbeta": fbeta,
    }
    return response_object
