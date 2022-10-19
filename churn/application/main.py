"""Main for churn modelling """
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from churn.infrastructure.bank_customers import BankCustomersData
from churn.domain.bank_customers_dataset import FeaturesDataset

np.random.seed(42)


F_INDICATORS = "./data/1_-_indicators.csv"
F_CUSTOMERS = "./data/1_-_customers.csv"
bcd = BankCustomersData(F_INDICATORS, F_CUSTOMERS)
raw_data = bcd.load_data()

# feature engineering
fds = FeaturesDataset(balance_imputation="drop")
features = fds.compute_features(raw_data)

# model performance
clf = DecisionTreeClassifier(max_depth=5)
X_train, X_test, y_train, y_test = train_test_split(
    features.drop(columns=["CHURN"]), features.CHURN, test_size=.15)
clf.fit(X_train, y_train)

print(f"accuracy scores:\n"
      f"\ttest:  {clf.score(X_test, y_test)}\n"
      f'\ttrain: {clf.score(X_train, y_train)}')
