import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, load_boston


def load_binary():
    data = load_breast_cancer(as_frame=True)
    return data["data"], data["target"]

def load_regression():
    data = load_boston()
    X = pd.DataFrame(data["data"], columns=data["feature_names"])
    y = pd.Series(data["target"], name="target")
    return X, y

def load_multiclass():
    return load_wine(as_frame=True, return_X_y=True)
