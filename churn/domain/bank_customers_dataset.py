"""This module compute features for churn detection"""
from dataclasses import dataclass
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.base import TransformerMixin

@dataclass
class FeaturesDataset(TransformerMixin):
    """This class represents the features of the churn modelling."""
    features: pd.DataFrame = None
    balance_imputation: str = "median"

    def fit(self, X, y=None):
        return self

    def transform(self, raw_data, y=None) -> pd.DataFrame:
        """Compute features for churn detection.
        Args:
            raw_data (pd.DataFrame): output of BankCustomersData.load_data()
        """
        self.features = raw_data.drop(columns=["NOM"])\
                                .assign(NUM_DAYS=raw_data.DATE_ENTREE.apply(
                                    lambda x: (raw_data.DATE_ENTREE.max()-x).days))\
                                .assign(DAY_OF_YEAR=raw_data.DATE_ENTREE.dt.dayofyear)\
                                .drop(columns=["DATE_ENTREE"]).dropna()
        #FIXME dropna
        self._balance_imputation()
        self._encode_lands()

        return self.features

    def _balance_imputation(self):
        """impute value in zero in balance """
        assert self.balance_imputation in {"median", "mean", "none", "drop"}
        zeros_pos = self.features.BALANCE == 0

        imput_value = self.features.loc[~zeros_pos, "BALANCE"].median()
        if self.balance_imputation == "mean":
            imput_value = self.features.loc[~zeros_pos, "BALANCE"].mean()

        if self.balance_imputation != "none":
            self.features.loc[zeros_pos, "BALANCE"] = imput_value

        if self.balance_imputation == "drop":
            self.features = self.features.loc[~zeros_pos]


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
