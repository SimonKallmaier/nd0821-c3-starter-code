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
def model_fixture():
    return train_model(X_train, y_train)


@pytest.fixture
def inference_fixture(model_fixture):
    
    return inference(model=model_fixture, X=X_test)


@pytest.fixture
def metrics_fixture(inference_fixture):
    return compute_model_metrics(y=y_test, preds=inference_fixture)


def test_model(model_fixture):
    assert isinstance(model_fixture, LogisticRegression)


def test_inference(inference_fixture):
    assert isinstance(inference_fixture, np.ndarray)


def test_metrics(metrics_fixture):
    assert isinstance(metrics_fixture, tuple)
