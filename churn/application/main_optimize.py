import argparse
from statistics import mode
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
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from churn.domain.churn_model import ChurnModelSelection
from churn.domain.domain_utils import get_train_test_split
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
all_models_param = transform_to_object("grid_search_params",MODELS_MAPPING_DICT)

all_models = list()
# Function list_all_models, to create in param
for model in all_models_param:
    all_models.append(model['pipeline__classifier'][0])


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--models",
                help=f"Optimise through all models disponible in config file list. Ex : {all_models}",
                action="store_true")
parser.add_argument("-g", "--GridSearchModel", type=str,
                    help=("Compute a gridSearch for a specific model with all parameters defined in the config file."))
args = parser.parse_args()

bcd = BankCustomersData(CONFIG_DATA_INDICATORS, CONFIG_DATA_CUSTOMERS)
raw_data = bcd.load_data()
feature = FeaturesDataset().transform(raw_data)
#Waiting for the solution @Charles.
X_train, X_test, y_train, y_test = train_test_split(feature.drop(columns=["CHURN"]), feature["CHURN"], test_size=0.95, random_state=33)
#X_train, X_test, y_train, y_test = train_test_split(raw_data.drop(columns=["CHURN"]), raw_data["CHURN"], test_size=0.80, random_state=33)
print(type(all_models_param[0]["pipeline__classifier__gamma"][0]))
#all_models_param[0]["pipeline__classifier__gamma"] = (1e-6, 1e+1, 'log-uniform')
print(all_models_param[0]["pipeline__classifier__gamma"])
#Si l'utilisateur veut visualiser les scores de chacun des modèles avec leurs paramètres par défaut. 
if args.models:
    for model_classifier in all_models:
        print(f"Modèle en cours d'entrainement : {model_classifier}")
        model = ChurnModelSelection(pipeline=Pipeline([('scaler', StandardScaler()),('classifier', model_classifier)]))
        param = {}
        clf = GridSearchCV(model,param_grid=param)
        clf.fit(X_train,y_train)
        print(clf.best_score_)

    
if args.GridSearchModel:
    #Mettre ça dans un utils
    for model_parameters in all_models_param:
        classifier = model_parameters["pipeline__classifier"][0]
        if f"{classifier.__class__.__name__}()" == args.GridSearchModel :
            print("Run this model, this is the right one")
            #TODO : Remettre en place l'appel à FeaturesDataset.             
            model = ChurnModelSelection(pipeline=Pipeline([ ('scaler', StandardScaler()),('classifier', classifier)]))
            opt = BayesSearchCV(
                model,
                [(model_parameters, 5)],
                #Change cv to 5
                cv=2 )
            opt.fit(X_train, y_train)
            print("val. score: %s" % opt.best_score_)
            print("test score: %s" % opt.score(X_test, y_test))
            print(f"best params : {opt.best_params_}")

exit()

