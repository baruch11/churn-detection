import argparse, os
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from churn.domain.churn_model import ChurnModelSelection
from churn.domain.domain_utils import get_train_test_split, return_models_from_all_model_params, find_model_params_from_model_name
from churn.domain.bank_customers_dataset import FeaturesDataset
from churn.config.config import read_yaml, transform_to_object, save_best_params_to_yaml
from churn.domain.domain_utils import get_train_test_split
from churn.infrastructure.bank_customers import BankCustomersData
ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
np.random.seed(42)
#TODO : put it in config.py file
MODELS_MAPPING_DICT={
    "SVC()" : SVC(),
    "DecisionTreeClassifier()" : DecisionTreeClassifier(),
    "RandomForestClassifier()" : RandomForestClassifier(),
    "MLPClassifier()" : MLPClassifier(),
    "GradientBoostingClassifier()" : GradientBoostingClassifier(),
    "ExplainableBoostingClassifier()" : ExplainableBoostingClassifier()
}
CONFIG = read_yaml("churn/config/config_template.yml")
# TODO : Mettre ici un argparse avec un chemin par défaut. 
CONFIG_DATA_INDICATORS = CONFIG["data"]["indicators_dataset"]
CONFIG_DATA_CUSTOMERS = CONFIG["data"]["customers_dataset"]
all_models_param = transform_to_object("churn/config/config_template.yml","grid_search_params",MODELS_MAPPING_DICT)

all_models = return_models_from_all_model_params(all_models_param)

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--models",
                help=f"Optimise through all models disponible in config file list. Ex : {all_models}",
                action="store_true")
parser.add_argument("-g", "--GridSearchModel", type=str,
                    help=("Compute a gridSearch for a specific model with all parameters defined in the config file."))
parser.add_argument("-n", "--n_iter", type=str,
                    help=("(Work only for GridSearchModel) : Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution. Default is 5"),
                    default=5)




args = parser.parse_args()


bcd = BankCustomersData(CONFIG_DATA_INDICATORS, CONFIG_DATA_CUSTOMERS)
raw_data = bcd.load_data()
feature = FeaturesDataset().transform(raw_data)
# TODO : Mettre les raw data dans le pipeline
# TODO : Décommenter les lignes du fichier de config
# TODO : Enlever le test size = 0.95
#Waiting for the solution @Charles.
X_train, X_test, y_train, y_test = train_test_split(feature.drop(columns=["CHURN"]), feature["CHURN"], test_size=0.20, random_state=33)
#X_train, X_test, y_train, y_test = get_train_test_split()

#If the user want to visualize the score of each one of the different models. 
if args.models:
    for model_classifier in all_models:
        print(f"Modèle en cours d'entrainement : {model_classifier}")
        model = ChurnModelSelection(pipe=Pipeline([('scaler', StandardScaler()),('classifier', model_classifier)]))
        param = {}
        clf = GridSearchCV(model,param_grid=param)
        clf.fit(X_train,y_train)
        print(clf.best_estimator_.score_details(X_test,y_test))

#If the user want to commpute a Bayesian Search Optimization for a specific model
if args.GridSearchModel:
    model_parameters = find_model_params_from_model_name(all_models_param,model_name=args.GridSearchModel)
    print("model parameters : \n",model_parameters)
    classifier = model_parameters["pipe__classifier"][0]      
    model = ChurnModelSelection(pipe=Pipeline([ ('scaler', StandardScaler()),('classifier', classifier)]))
    
    opt = BayesSearchCV(
        model,
        [(model_parameters, int(args.n_iter))],
        cv=5 )
    opt.fit(X_train, y_train)
    print(opt.best_estimator_.score_details(X_test,y_test))
    print(f"best params : {opt.best_params_}")
    save_best_params_to_yaml(path="churn/config/latest_model.yml",best_params=opt.best_params_,model_name=f"{classifier.__class__.__name__}()")

exit()

