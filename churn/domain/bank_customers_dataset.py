"""This module compute features for churn detection"""
from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator


@dataclass
class FeaturesDataset(TransformerMixin, BaseEstimator):
    """This class represents the features of the churn modelling."""

    features: pd.DataFrame = None
    balance_imputation: str = "median"
    imput_nan_salaire = None
    imput_nan_score_credit = None
    imput_outliers_age = None
    imput_nan_balance = None
    imput_zero_balance = None
    land_encoder = OneHotEncoder(handle_unknown='ignore')

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the feature engineering transformer.
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

        self.land_encoder.fit(X[["PAYS"]])

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

        assert self.balance_imputation in {"median", "mean", "none"}

        self.features = raw_data.drop(columns=["NOM"])\
            .assign(NUM_DAYS=raw_data.DATE_ENTREE.apply(
                lambda x: (raw_data.DATE_ENTREE.max()-x).days))\
            .assign(DAY_OF_YEAR=raw_data.DATE_ENTREE.dt.dayofyear)\
            .drop(columns=["DATE_ENTREE"])

        outliers_pos = self.features["AGE"] > 100
        zeros_pos = self.features["BALANCE"] == 0

        self._encode_lands()
        self.features["SALAIRE"].fillna(self.imput_nan_salaire, inplace=True)
        self.features["SCORE_CREDIT"].fillna(
            self.imput_nan_score_credit, inplace=True)
        self.features["BALANCE"].fillna(self.imput_nan_balance, inplace=True)
        self.features.loc[outliers_pos, "AGE"] = self.imput_outliers_age
        if self.balance_imputation != "none":
            self.features.loc[zeros_pos, "BALANCE"] = self.imput_zero_balance

        return self.features

    # ENCODING

    def _encode_lands(self):
        """Onehot encoding of 'PAYS' feature."""
        oh_pays = pd.DataFrame(
            self.land_encoder.transform(self.features[["PAYS"]]).todense(),
            columns=self.land_encoder.categories_[0],
            index=self.features.index).astype(bool)
        self.features = self.features.drop(columns="PAYS")\
                                     .join(oh_pays)
