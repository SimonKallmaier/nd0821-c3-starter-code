import requests
import json

data = {
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

r = requests.post("http://127.0.0.1:8000/predict/", data=json.dumps(data))

print(r.json())
