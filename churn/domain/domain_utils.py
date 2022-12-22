"""functions related to domain"""
import opcode
import os

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from churn.infrastructure.bank_customers import BankCustomersData



def get_rootdir():
    """Return rootdir absolute path"""
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../"))


def get_train_test_split(save_test=False):
    """Return dataset split used in application
    the dataset files path is hard coded <rootdir>/churn/config
    Returns:
    --------
        X_train (DataFrame)
        X_test (DataFrame),
        y_train (Series)
        y_test (Series)
    """
    param_file = os.path.join(get_rootdir(),
                              "churn/config/config_template.yml")
    with open(param_file, "r", encoding="utf-8") as fparam:
        cfg_dict = yaml.load(fparam, Loader=yaml.FullLoader)

    indicators_path = os.path.join(get_rootdir(),
                                   cfg_dict["data"]["indicators_dataset"])
    customers_path = os.path.join(get_rootdir(),
                                  cfg_dict["data"]["customers_dataset"])

    bcd = BankCustomersData(indicators_path, customers_path)
    raw_data = bcd.load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        raw_data.drop(columns=["CHURN"]),
        raw_data["CHURN"] == "Yes",
        test_size=0.20,
        random_state=33)

    if save_test:
        pd.concat([X_test, y_test.rename("label")],
                  axis=1).to_csv(_test_set_path())

    return X_train, X_test, y_train, y_test


def return_models_from_all_model_params(all_models_param : dict) -> list:
    """ Return all models for a list of param grid"""
    all_models = list()
    for model in all_models_param:
        all_models.append(model['pipe__classifier'][0])
    return all_models

def find_model_params_from_model_name(all_models_param : dict,model_name : str) -> dict:
    """Find specific model param from the model name."""
    right_model_params = None
    for model_parameters in all_models_param:
        classifier = model_parameters["pipe__classifier"][0]
        if f"{classifier.__class__.__name__}()" == model_name :
            right_model_params = model_parameters
    if right_model_params is None:
        raise Exception("No model matches your search")  
    return right_model_params


def _test_set_path():
    """Return test set path."""
    this_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(this_dir, "test_dataset.csv")

def get_test_set():
    """Standalone function to load test set for test purpose.

    test_dataset.csv is produced by get_train_test_split

    Returns
    -------
    X_test : pd.DataFrame
    y_test : pd.Series
    """
    test_dataset = pd.read_csv(_test_set_path())
    X_test = test_dataset.drop(columns=["label"])
    y_test = test_dataset["label"]
    return X_test, y_test
