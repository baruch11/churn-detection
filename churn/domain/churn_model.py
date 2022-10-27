import os
from abc import ABC, ABCMeta, abstractmethod
from xmlrpc.client import Boolean
import pandas as pd
from sklearn.pipeline import Pipeline
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from churn.domain.bank_customers_dataset import FeaturesDataset
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             precision_score)
from churn.domain.bank_customers_dataset import FeaturesDataset


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

    def score_details(self, X_test, y_test):
        """Return a dataframe with multiple metrics on several subsets
        Returns
        -------
           pd.Dataframe: columns: metrics, index: subsets
        """
        m_test = pd.concat([X_test,
                            self.predict(X_test).rename("pred"),
                            y_test], axis=1)

        def _my_scores(df):
            metrics = [accuracy_score, f1_score, precision_score, recall_score]
            ret = {metric.__name__: metric(df.CHURN, df.pred)
                   for metric in metrics}
            return pd.Series(ret)

        scores_pays = m_test.groupby("PAYS").apply(_my_scores)

        balance0 = (X_test.BALANCE == 0)
        scores_balance = m_test.groupby(balance0)\
                               .apply(_my_scores)\
                               .rename(index={True: "balance = 0",
                                              False: "balance > 0"})
        score_total = pd.DataFrame(
            [_my_scores(m_test).to_dict()]).rename(index={0: "global"})

        return pd.concat([score_total, scores_pays, scores_balance])



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



class ChurnModelFinal(BaseChurnModel):

    def __init__(self, _max_depth=5):
        self.pipe = Pipeline([
            ('features', FeaturesDataset()),
            ('clf', DecisionTreeClassifier(max_depth=_max_depth))
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.pipe.fit(X, y)

    def predict(self, X: pd.DataFrame):
        return pd.Series(
            self.pipe.predict(X),
            index = X.index
        )

class ChurnModelSelection(BaseChurnModel,BaseEstimator, ClassifierMixin):
    def __init__(self,pipeline : Pipeline):
        self.pipeline = pipeline
    def fit(self,X : pd.DataFrame, y : pd.DataFrame):

        """Build the models of the given pipeline from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples taken from raw data (infrastructure output).

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.
        """
        print(f"Pipeline Get params : {self.pipeline. get_params(deep=True)}")
        
        #fds = FeaturesDataset(balance_imputation=self.balance_imputation)
        #X,y = fds.compute_features(X),fds.compute_features(y)
        self.pipeline.fit(X,y)
        #X, y = check_X_y(X, y, accept_sparse=True)
        #X, y = check_X_y(X, y)
        return self
    def score(self,X,y):
        score = self.pipeline.score(X,y)
        return  score
    def predict(self,X):
        y_hat = self.pipeline.predict(X)
        #check_is_fitted(self)
        return y_hat
