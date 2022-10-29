"""Unit test for FeatureDataset"""
import os
import numpy as np
import pandas as pd

from churn.infrastructure.bank_customers import BankCustomersData
from churn.domain.bank_customers_dataset import FeaturesDataset

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))


def test_compute_features():
    bcd = BankCustomersData(os.path.join(ROOTDIR, "data/1_-_indicators.csv"),
                            os.path.join(ROOTDIR, "data/1_-_customers.csv"))
    raw_data = bcd.load_data().drop(columns=["CHURN"])
    fds = FeaturesDataset()

    features = fds.fit_transform(raw_data)

    columns = fds.get_feature_names_out()
    df_features = pd.DataFrame(features, columns=columns, index=raw_data.index)

    print(df_features.head(2).T)
    print("Isnull :\n", df_features.isna().any())

    assert not df_features.isna().any().any()
