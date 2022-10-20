from abc import ABC, ABCMeta, abstractmethod
from xmlrpc.client import Boolean
import pandas as pd
from sklearn.pipeline import Pipeline
import pickle

class BaseChurnModel(metaclass=ABCMeta):
    """
    Interface of our ChurnModel

    """
    
    @abstractmethod
    def fit(self):
        raise NotImplementedError
    @abstractmethod
    def predict(self):
        raise NotImplementedError
    @abstractmethod
    def load(self):
        raise NotImplementedError
    @abstractmethod
    def save(self):
        raise NotImplementedError

class DummyChurnModel(BaseChurnModel):
    PICKLE_PATH ="churn/config/DummyChurnModel.pkl"
    def __init__(self,params : dict = None) :
        self.params = params or {}

    def fit(self,X : pd.DataFrame, y :pd.DataFrame, pipeline : Pipeline) -> 'DummyChurnModel' :
        """Build the models of the given pipeline from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. 

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.

        pipeline : 
            Pipeline of transforms with a final estimator.

        Returns
        -------
        self : DummyChurnModel
            Fitted pipeline.
        """
        self.pipeline = pipeline.fit(X, y)
        return self
    def predict(self,X : pd.DataFrame):
        predict =self.pipeline.predict(X)
        return predict
    @staticmethod
    def load() -> Boolean:
        with open(str(DummyChurnModel.PICKLE_PATH), 'rb') as inp:
            return pickle.load(inp)
    def save(self) -> Boolean:
        with open(str(self.PICKLE_PATH), 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
        return True 