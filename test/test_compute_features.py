"""Unit test for FeatureDataset"""
import os

from churn.infrastructure.bank_customers import BankCustomersData
from churn.domain.bank_customers_dataset import FeaturesDataset

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))


def test_compute_features():
    bcd = BankCustomersData(os.path.join(ROOTDIR, "data/1_-_indicators.csv"),
                            os.path.join(ROOTDIR, "data/1_-_customers.csv"))
    raw_data = bcd.load_data()
    fds = FeaturesDataset()

    features = fds.fit_transform(raw_data)

    assert not features.isnull().any().any()
