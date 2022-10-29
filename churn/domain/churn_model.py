"""Model for churn detection"""
import os
import pickle

from abc import ABCMeta, abstractmethod
from xmlrpc.client import Boolean
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             precision_score)

from churn.domain.bank_customers_dataset import FeaturesDataset
from churn.config.config import retrieve_optimal_parameters



class BaseChurnModel(metaclass=ABCMeta):
    """
    Interface of our ChurnModel

    """
    PICKLE_ROOT = "data/models"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the model from the training set (X, y)."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame):
        """Return predictions."""
        raise NotImplementedError

    @classmethod
    def load(cls) -> Boolean:
        """Load the churn model from the class instance name."""
        src_dir = os.path.join(cls.PICKLE_ROOT, cls.__name__) + ".pkl"
        with open(src_dir, 'rb') as inp:
            return pickle.load(inp)

    def save(self) -> Boolean:
        """Save the model (pkl format)."""
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
    """This class is a dummy model possibly used for a quick sanity check."""
    def __init__(self):
        self.clf = DecisionTreeClassifier()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the model from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples taken from raw data (infrastructure output).

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.
        """
        self.clf.fit(self._feature_engineering(X), y)

    def predict(self, X: pd.DataFrame):
        """Return predictions."""
        return pd.Series(
            self.clf.predict(self._feature_engineering(X)),
            index=X.index
        )

    def _feature_engineering(self, X):
        return X[["AGE"]]


class ChurnModelFinal(BaseChurnModel, BaseEstimator):
    """This class represents the final model for churn detection."""
    def __init__(self):
        self._accessing_optimal_model()
        self.pipe = Pipeline([
            ('features', FeaturesDataset()),
            ('clf', self.model_final)
        ])
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the model from the training set (X, y).

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The training input samples taken from raw data (infrastructure output).

        y : array-like of shape (n_samples,1)
            The target values (class labels) of booleans (churn or not churn).
        """
        self.pipe.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        """Make predictions.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The training input samples taken from raw data (infrastructure output).

        Returns
        -------
        y : pd.Series(boolean) of length n_samples, indexed like X
            result of the model prediction(churn or not churn).
        """
        return pd.Series(
            self.pipe.predict(X),
            index=X.index
        )

    def _accessing_optimal_model(self):
        #Access the chosen model
        self.model_final = retrieve_optimal_parameters()[0]



        

class ChurnModelSelection(BaseChurnModel, BaseEstimator, ClassifierMixin):
    """This class is a customizable churnmodel used for hyperparameters optimization"""
    def __init__(self, pipe: Pipeline):
        self.pipe = pipe

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Build the models of the given pipeline from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples taken from raw data (infrastructure output).

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.
        """

        self.pipe.fit(X, y)
        return self

    def score(self, X, y, sample_weight=None):
        score = self.pipe.score(X, y)
        return score

    def predict(self, X):
        return pd.Series(
            self.pipe.predict(X),
            index=X.index
        )
