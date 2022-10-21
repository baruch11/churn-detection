import os
from abc import ABC, ABCMeta, abstractmethod
from xmlrpc.client import Boolean
import pandas as pd
from sklearn.pipeline import Pipeline
import pickle
from sklearn.tree import DecisionTreeClassifier

class BaseChurnModel(metaclass=ABCMeta):
    """
    Interface of our ChurnModel

    """
    PICKLE_ROOT = "data/models"

    @abstractmethod
    def fit(self):
        raise NotImplementedError
    @abstractmethod
    def predict(self):
        raise NotImplementedError

    @classmethod
    def load(self) -> Boolean:
        src_dir = os.path.join(self.PICKLE_ROOT, self.__name__) + ".pkl"
        with open(src_dir, 'rb') as inp:
            return pickle.load(inp)

    def save(self) -> Boolean:
        with open(self._get_pickle_path(), 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
        return True

    def _get_pickle_path(self):
        return os.path.join(
            self.PICKLE_ROOT, self.__class__.__name__)+".pkl"


class DummyChurnModel(BaseChurnModel):

    def __init__(self):
        self.clf = DecisionTreeClassifier()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Build the models of the given pipeline from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples taken from raw data (infrastructure output).

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.
        """
        self.clf.fit(self._feature_engineering(X), y)

    def predict(self, X: pd.DataFrame):
        return pd.Series(
            self.clf.predict(self._feature_engineering(X)),
            index = X.index
        )


    def _feature_engineering(self, X):
        return X[["AGE"]]
