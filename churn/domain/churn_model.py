from abc import ABC, ABCMeta, abstractmethod
from xmlrpc.client import Boolean
import pandas as pd
from sklearn.pipeline import Pipeline



class BaseChurnModel(metaclass=ABCMeta):
    """
    Interface of our ChurnModel

    """
    
    @abstractmethod
    def fit(self):
        raise NotImplementedError
    def predict(self):
        raise NotImplementedError
    @abstractmethod
    def load(self):
        raise NotImplementedError
    @abstractmethod
    def save(self):
        raise NotImplementedError

class DummyChurnModel(BaseChurnModel):
    def __init__(self,X : pd.DataFrame = None, y :pd.DataFrame = None, pipeline : Pipeline = None, params : dict = None) :
        self.X = X 
        self.y = y 
        self.pipeline = pipeline
        self.params = dict or {}
    def fit(self) -> 'DummyChurnModel' :
        return self
    def predict(self):
        return self.y
    def load(self) -> Boolean:
        return True
    def save(self) -> Boolean:
        return True 