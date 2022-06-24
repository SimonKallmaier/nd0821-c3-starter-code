# Script to train machine learning model.
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv(os.path.join("starter", "data", "census_cleaned.csv"))


def process_data(data, categorical_features, label, training):

    num_features = [
        "age", "education-num", "capital-gain", "capital-loss", "hours-per-week"
    ]
    encoder = OneHotEncoder().fit(data[categorical_features])
    one_hot_encoded_cat_vars = encoder.transform(data[categorical_features])
    
    X_array = np.concatenate([one_hot_encoded_cat_vars.A, data[num_features].to_numpy()], axis=1)
    
    y = (data[label] == ">50K").to_numpy().reshape(-1, 1)
    
    return X_array, y, encoder


# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(train, categorical_features=cat_features, label="salary", training=True)

# Proces the test data with the process_data function.

# Train and save a model.
