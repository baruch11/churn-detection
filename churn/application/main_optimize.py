"""Main for churn modelling """
from curses import raw
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from churn.domain.churn_model import ChurnModelSelection
from churn.infrastructure.bank_customers import BankCustomersData
from churn.domain.bank_customers_dataset import FeaturesDataset
from churn.config.config import read_yaml

np.random.seed(42)

CONFIG = read_yaml("churn/config/config_template.yml")

CONFIG_DATA_INDICATORS = CONFIG["data"]["indicators_dataset"]
CONFIG_DATA_CUSTOMERS = CONFIG["data"]["customers_dataset"]
bcd = BankCustomersData(CONFIG_DATA_INDICATORS, CONFIG_DATA_CUSTOMERS)
raw_data = bcd.load_data()
raw_data = raw_data[0:100]
#TODO : Faire une feature request pour eque l'objet FeatureDataset puisse fonctionner sur X et sur y, sinon on ne pourra jamais faire un ChurnModel.fit(feature.drop('CHURN'),feature["CHURN"])
fds = FeaturesDataset(balance_imputation="drop")
features = fds.compute_features(raw_data)
# feature engineering
estimator = ChurnModelSelection(pipeline=Pipeline([('scaler', StandardScaler()), ('svc', SVC())]))
#TODO : Faire varier balance imputation
params={'pipeline__svc__C':[.01,.05,.1,.5,1,5,10]}
clf = GridSearchCV(estimator, params)
clf.fit(X=features.drop(columns=["CHURN"]),y=features["CHURN"])
print(clf.cv_results_)