import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from churn.domain.churn_model import ChurnModelSelection
from churn.infrastructure.bank_customers import BankCustomersData
from churn.domain.bank_customers_dataset import FeaturesDataset
from churn.config.config import read_yaml, transform_to_object

np.random.seed(42)
MODELS_MAPPING_DICT={
    "SVC()" : SVC(),
    "DecisionTreeClassifier()" : DecisionTreeClassifier(),
    "RandomForestClassifier()" : RandomForestClassifier(),
    "AdaBoostClassifier()" : AdaBoostClassifier(),
    "MLPClassifier()" : MLPClassifier()
}
CONFIG = read_yaml("churn/config/config_template.yml")

CONFIG_DATA_INDICATORS = CONFIG["data"]["indicators_dataset"]
CONFIG_DATA_CUSTOMERS = CONFIG["data"]["customers_dataset"]
models = transform_to_object("grid_search_params",MODELS_MAPPING_DICT)
#MAPPING_DICT[CONFIG_MODEL_PARAMETERS[1]["pipeline__classifier"][0]]

print(type(models[0]["pipeline__classifier"][0]))

bcd = BankCustomersData(CONFIG_DATA_INDICATORS, CONFIG_DATA_CUSTOMERS)
raw_data = bcd.load_data()
# TODO : Faire un for pour boucler sur chaque modèle, et exporter les résultats _best_params, _best_score pour chacun d'eux. 
X_train, X_test, y_train, y_test = train_test_split(raw_data.drop(columns=["CHURN"]), raw_data["CHURN"], test_size=0.20, random_state=33)
model = ChurnModelSelection(pipeline=Pipeline([('features', FeaturesDataset()), ('classifier', first_estimator)]))
clf = GridSearchCV(model, first_grid_params)
clf.fit(X_train,y_train)
print(clf.best_params_)
print(clf.score(X_test,y_test))