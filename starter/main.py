# Put the code for your API here.

from fastapi import FastAPI
from pydantic import BaseModel

from starter.ml.model import inference

app = FastAPI()


class Data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


@app.get("/")
async def say_hello():
    return {"greeting": "Hello!"}



@app.post("/model")
async def exercise_function(path: int, query: int, body: Value):
    return {"path": path, "query": query, "body": body}


@app.post("/predict", response_model=Data, status_code=200)
def get_prediction(payload: Data):


    prediction = inference(ticker)

    response_object = {"prediction": convert(prediction_list)}
    return response_object
