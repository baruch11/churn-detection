"""Unit test for """
from churn.domain.domain_utils import get_train_test_split

import logging

def test_split():
    X_train, X_test, y_train, y_test = get_train_test_split()
    # check if the split is stratified
    acceptance_thres = .02
    # PAYS
    train_split = X_train["PAYS"].value_counts(normalize=True)
    test_split = X_test["PAYS"].value_counts(normalize=True)
    logging.info("PAYS:\ntrain %s\ntest%s", train_split, test_split)
    assert not ((train_split - test_split).abs() > acceptance_thres).any()
    # BLANCE 0
    train_split = (X_train.BALANCE == 0).mean()
    test_split = (X_test.BALANCE == 0).mean()
    logging.info("BALANCE 0:\ntrain %s\ntest%s", train_split, test_split)
    assert (train_split-test_split) < acceptance_thres
