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
        
    def fit(self, X:pd.DataFrame, y=None):
        
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

        
        return self

    def transform(self, X_processed:pd.DataFrame, y=None) -> pd.DataFrame:
        """Compute features for churn detection.
        Args:
            raw_data (pd.DataFrame): output of BankCustomersData.load_data()
        """

        assert self.balance_imputation in {"median", "mean", "none"}
       
        self.features = X_processed.drop(columns=["NOM"])\
                                .assign(NUM_DAYS=X_processed.DATE_ENTREE.apply(
                                    lambda x: (X_processed.DATE_ENTREE.max()-x).days))\
                                .assign(DAY_OF_YEAR=X_processed.DATE_ENTREE.dt.dayofyear)\
                                .drop(columns=["DATE_ENTREE"])
        
        outliers_pos = self.features["AGE"] > 100
        zeros_pos = self.features["BALANCE"] == 0
        
        self._encode_lands()
        self.features["SALAIRE"].fillna(self.imput_nan_salaire, inplace = True)
        self.features["SCORE_CREDIT"].fillna(self.imput_nan_score_credit, inplace = True)
        self.features["BALANCE"].fillna(self.imput_nan_balance, inplace = True)
        self.features.loc[outliers_pos, "AGE"] = self.imput_outliers_age
        if self.balance_imputation != "none":
            self.features.loc[zeros_pos, "BALANCE"] = self.imput_zero_balance

        
        return self.features

    
    #ENCODING
    def _encode_lands(self):
        """Onehot encoding of 'PAYS' feature."""
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(self.features[["PAYS"]])
        oh_pays = pd.DataFrame(
            enc.transform(self.features[["PAYS"]]).todense(),
            columns=enc.categories_[0],
            index=self.features.index).astype(bool)
        self.features = self.features.drop(columns="PAYS")\
                                     .join(oh_pays)