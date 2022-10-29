"""Model for churn detection"""
import os
import pickle

from abc import ABCMeta, abstractmethod
from xmlrpc.client import Boolean
import pandas as pd

from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             precision_score)

from churn.domain.bank_customers_dataset import FeaturesDataset
from churn.config.config import read_yaml, transform_to_object
from churn.domain.domain_utils import find_model_params_from_model_name



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


class ChurnModelFinal(BaseChurnModel):
    """This class represents the final model for churn detection."""
    def __init__(self, _max_depth=5):
        self._retrieve_optimal_parameters()
        self.pipe = Pipeline([
            ('features', FeaturesDataset()),
            ('clf', ExplainableBoostingClassifier(binning=self.optimal_binning,
             early_stopping_tolerance=self.optimal_early_stopping_tolerance,
             learning_rate=self.optimal_learning_rate,
              min_samples_leaf=self.optimal_min_samples_leaf, 
              outer_bags= self.optimal_outer_bags))
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

    def _retrieve_optimal_parameters(self):

        ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        CONFIG = read_yaml(os.path.join(ROOTDIR, "churn/config/latest_model.yml"))
        model_final_params_list = transform_to_object("churn/config/latest_model.yml","model_parameters")
        model_final_params_dict = find_model_params_from_model_name(model_final_params_list, model_name="ExplainableBoostingClassifier()")
        self.optimal_binning = model_final_params_dict["pipe__classifier__binning"]
        self.optimal_early_stopping_tolerance = model_final_params_dict["pipe__classifier__early_stopping_tolerance"]
        self.optimal_learning_rate = model_final_params_dict["pipe__classifier__learning_rate"]
        self.optimal_min_samples_leaf = model_final_params_dict["pipe__classifier__min_samples_leaf"]
        self.optimal_outer_bags = model_final_params_dict["pipe__classifier__outer_bags"]




        

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
