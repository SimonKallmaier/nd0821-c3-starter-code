# Script to train machine learning model.
import os
import sys

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split


sys.path.append(os.path.join(os.getcwd(), "ml"))
print(sys.path)
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv(os.path.join("data", "census_cleaned.csv"))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=1)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(train, cat_features, label="salary", training=True)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(test, cat_features, label="salary", training=False, encoder=encoder, lb=lb)

# Train and save a model.
model = train_model(X_train, y_train)
joblib.dump(model, os.path.join("model", "model.joblib"))
joblib.dump(encoder, os.path.join("model", "encoder.joblib"))
joblib.dump(lb, os.path.join("model", "lb.joblib"))


# inference pipeline
y_pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y=y_test, preds=y_pred)


def inference_sliced_data(data: pd.DataFrame, category: str, value: str):
    """This function depends on global variables --> Bad practice"""
    X = data.loc[data[category] == value, :]
    X_test_sliced, y_test_sliced, _, _ = process_data(X, cat_features, label="salary", training=False, encoder=encoder, lb=lb)
    return compute_model_metrics(y=y_test_sliced, preds=inference(model, X_test_sliced))
