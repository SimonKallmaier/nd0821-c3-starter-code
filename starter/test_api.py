import json
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


data_prediction_0 = {
    "age": 30,
    "workclass": "Private",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Divorced",
    "occupation": "Sales",
    "relationship": "Unmarried",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
    "salary": "<=50K",
}
data_prediction_1 = {
    "age": 30,
    "workclass": "Private",
    "fnlgt": 77516,
    "education": "Doctorate",
    "education-num": 16,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 5178,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
    "salary": ">50K",
}


def test_get_path():
    r = client.get("")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello!"}


def test_post_0():
    data = json.dumps(data_prediction_0)
    r = client.post("/predict", data=data)
    assert r.status_code == 200
    prediction = r.json()
    assert prediction["prediction"] == [0]


def test_post_1():
    data = json.dumps(data_prediction_1)
    r = client.post("/predict", data=data)
    assert r.status_code == 200
    prediction = r.json()
    assert prediction["prediction"] == [1]
