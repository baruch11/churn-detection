"""functions related to domain"""
import os
import yaml

from sklearn.model_selection import train_test_split

from churn.infrastructure.bank_customers import BankCustomersData



def get_rootdir():
    """Return rootdir absolute path"""
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../"))


def get_train_test_split():
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
        raw_data["CHURN"],
        test_size=0.20,
        random_state=33)

    return X_train, X_test, y_train, y_test
