"""Main for churn modelling """
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from churn.domain.churn_model import DummyChurnModel
from churn.infrastructure.bank_customers import BankCustomersData
from churn.domain.bank_customers_dataset import FeaturesDataset
from churn.config.config import read_yaml

np.random.seed(42)

CONFIG = read_yaml("churn/config/config_template.yml")

CONFIG_DATA_INDICATORS = CONFIG["data"]["indicators_dataset"]
CONFIG_DATA_CUSTOMERS = CONFIG["data"]["customers_dataset"]
bcd = BankCustomersData(CONFIG_DATA_INDICATORS, CONFIG_DATA_CUSTOMERS)
raw_data = bcd.load_data()

# feature engineering
fds = FeaturesDataset(balance_imputation="drop")
features = fds.compute_features(raw_data)
dummy = DummyChurnModel(X=features.drop(columns=["CHURN"]),y=features["CHURN"])
print("dummy fiting :",dummy.fit())
print("dummy predicting :",dummy.predict())
print("dummy loading :",dummy.load())
print("dummy saving :",dummy.save())
# model performance
clf = DecisionTreeClassifier(max_depth=5)
X_train, X_test, y_train, y_test = train_test_split(
    features.drop(columns=["CHURN"]), features.CHURN, test_size=.15)
clf.fit(X_train, y_train)

print(f"accuracy scores:\n"
      f"\ttest:  {clf.score(X_test, y_test)}\n"
      f'\ttrain: {clf.score(X_train, y_train)}')
