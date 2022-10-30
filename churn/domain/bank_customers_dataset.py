"""This module compute features for churn detection"""
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer

def FeaturesDataset(balance_imputation='median',
                    balance_zero=True,
                    drop_cols=["NOM", "DATE_ENTREE"]):
    """This function return a preprocessing transformer for the churn model.
    Parameters
    ----------
    balance_imputation (str): balance 0 imputation mode (none|mean|median)
    balance_zero (bool): if True add a boolean column for BALANCE == 0
    drop_cols (list of str): list of column to drop
    """

    land_encoder = OneHotEncoder(handle_unknown='ignore', dtype=np.float64)
    return Pipeline(
        steps=[('imputer',
                FeaturesImputer(balance_imputation=balance_imputation,
                                balance_zero=balance_zero)),
               ('col_transform', make_column_transformer(
                   (land_encoder, ["PAYS"]),
                   ("drop", drop_cols),
                   remainder='passthrough'))
               ]
        )


@dataclass
class FeaturesImputer(TransformerMixin, BaseEstimator):
    """This class is an imputer for missing datas or outliers.
    Parameters
    ----------
    balance_imputation (str): balance 0 imputation mode (none|mean|median)
    balance_zero (bool): if True add a boolean column for BALANCE == 0
    """

    balance_imputation: str = "median"
    balance_zero: bool = False
   
    imput_nan_salaire = None
    imput_nan_score_credit = None
    imput_outliers_age = None
    imput_nan_balance = None
    imput_zero_balance = None
    feature_names = None

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the imputation transformer.
        Parameters
        ----------
        X (pd.DataFrame): raw data output of
             infrastructure/BankCustomersData.load_data()
        y: not used, sklearn API compatibility
        """
        assert self.balance_imputation in {"median", "mean", "none"}

        zeros_pos = X["BALANCE"] == 0
        outliers_pos = X["AGE"] > 100
        self.imput_nan_salaire = X["SALAIRE"].median()
        self.imput_nan_score_credit = X["SCORE_CREDIT"].median()
        self.imput_outliers_age = X.loc[~outliers_pos, "AGE"].median()
        self.imput_nan_balance = X["BALANCE"].median()

        if self.balance_imputation == "median":
            self.imput_zero_balance = X.loc[~zeros_pos, "BALANCE"].median()

        if self.balance_imputation == "mean":
            self.imput_zero_balance = X.loc[~zeros_pos, "BALANCE"].mean()

        self.feature_names = list(X.columns)
        if self.balance_zero:
            self.feature_names.append("balance_zero")

        return self

    def transform(self, raw_data: pd.DataFrame, y=None) -> pd.DataFrame:
        """Compute features for churn detection.
        Parameters
        ----------
        raw_data (pd.DataFrame): output of BankCustomersData.load_data()
        y: not used, sklearn API compatibility

        Returns
        -------
        features (pd.DataFrame): features matrix
        """

        features = raw_data

        outliers_pos = features["AGE"] > 100
        zeros_pos = features["BALANCE"] == 0

        features["SALAIRE"].fillna(self.imput_nan_salaire, inplace=True)
        features["SCORE_CREDIT"].fillna(
            self.imput_nan_score_credit, inplace=True)
        features["BALANCE"].fillna(self.imput_nan_balance, inplace=True)
        features.loc[outliers_pos, "AGE"] = self.imput_outliers_age
        if self.balance_imputation != "none":
            features.loc[zeros_pos, "BALANCE"] = self.imput_zero_balance

        if self.balance_zero:
            features['balance_zero'] = zeros_pos

        return features


    def get_feature_names_out(self, input_features=None):
        return self.feature_names
