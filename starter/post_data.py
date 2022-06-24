import requests
import json

data = {
    "age": 30,
    "workclass": "Private",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Divorced",
    "occupation": "Sales",
    "relationship": "Unmarried",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States",
    "salary": ">50K"
}

r = requests.post("http://127.0.0.1:8000/predict/", data=json.dumps(data))

print(r.json())
