import pytest
import numpy as np

from sklearn.linear_model._logistic import LogisticRegression

from starter.starter.ml.model import train_model, compute_model_metrics, inference

train_size = 10
test_size = 4
nb_feature = 4

X_train = np.random.randint(5, size=(train_size, nb_feature))
y_train = np.random.randint(2, size=train_size).ravel()

X_test = np.random.randint(5, size=(test_size, nb_feature))
y_test = np.random.randint(2, size=test_size).ravel()


@pytest.fixture
def model():
    return train_model(X_train, y_train)


@pytest.fixture
def inference(model):
    return inference(model=model, X=X_test)


@pytest.fixture
def metrics(inference):
    return compute_model_metrics(y=y_test, preds=inference)


def test_model(model):
    assert isinstance(model, LogisticRegression)


def test_inference(inference):
    assert isinstance(inference, np.ndarray)


def test_metrics(metrics):
    assert isinstance(metrics, tuple)
