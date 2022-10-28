import argparse, os, logging
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from churn.domain.churn_model import ChurnModelSelection
from churn.domain.domain_utils import get_train_test_split, return_models_from_all_model_params, find_model_params_from_model_name
from churn.domain.bank_customers_dataset import FeaturesDataset
from churn.config.config import read_yaml, transform_to_object, save_best_params_to_yaml
from churn.domain.domain_utils import get_train_test_split
from churn.infrastructure.bank_customers import BankCustomersData
ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
CONFIG = read_yaml(os.path.join(ROOTDIR, "churn/config/config_template.yml"))
CONFIG_DATA_INDICATORS = CONFIG["data"]["indicators_dataset"]
CONFIG_DATA_CUSTOMERS = CONFIG["data"]["customers_dataset"]
all_models_param = transform_to_object("churn/config/config_template.yml","grid_search_params")

all_models = return_models_from_all_model_params(all_models_param)

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--models",
                help=f"Optimise through all models disponible in config file list. Ex : {all_models}",
                action="store_true")
parser.add_argument("-b", "--BayesSearchCV", type=str,
                    help="Compute a BayesSearchCV for a specific model with all parameters defined in the config file.")
parser.add_argument("-n", "--n_iter", type=str,
                    help="(Work only for BayesSearchCV) : Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution. Default is 5",
                    default=5)
parser.add_argument('--debug', '-d', action='store_true',
                    help='activate debug logs')
args = parser.parse_args()

logging.getLogger().setLevel(logging.INFO)
if args.debug:
    logging.getLogger().setLevel(logging.DEBUG)


logging.debug("Loading data...")
X_train, X_test, y_train, y_test = get_train_test_split()

#If the user want to visualize the score of each one of the different models. 
if args.models:
    logging.debug("Fitting on all models, with default parameters.")
    for model_classifier in all_models:
        logging.info(f"Training model {model_classifier}")
        model = ChurnModelSelection(pipe=Pipeline([('features', FeaturesDataset()),('scaler', StandardScaler()),('classifier', model_classifier)]))
        param = {}
        clf = GridSearchCV(model,param_grid=param)
        clf.fit(X_train, y_train)
        score_details = clf.best_estimator_.score_details(X_test,y_test)
        logging.info(f"Scores on Test dataset : \n {score_details.loc[['global']]}")
        logging.debug(f"Detailed scores on Test dataset : \n {score_details}")

#If the user want to commpute a Bayesian Search Optimization for a specific model
if args.BayesSearchCV:
    model_parameters = find_model_params_from_model_name(all_models_param,model_name=args.BayesSearchCV)
    classifier = model_parameters["pipe__classifier"][0]      
    logging.info(f"Computing BayesSearchCV for model :\n {classifier}")
    logging.debug(f"Model parameters are :\n {model_parameters}")
    model = ChurnModelSelection(pipe=Pipeline([('features',FeaturesDataset()),('scaler', StandardScaler()),('classifier', classifier)]))
    opt = BayesSearchCV(
        model,
        [(model_parameters, int(args.n_iter))],
        cv=5 )
    opt.fit(X_train, y_train)
    score_details = opt.best_estimator_.score_details(X_test,y_test)
    logging.info(f"Scores on Test dataset : \n {score_details.loc[['global']]}")
    logging.debug(f"Detailed scores on Test dataset : \n {score_details}")
    logging.info(f"Best parameters find are : {opt.best_params_}")
    logging.debug("Saving those parameters to latest_model.yml")
    save_best_params_to_yaml(path="churn/config/latest_model.yml",best_params=opt.best_params_,model_name=f"{classifier.__class__.__name__}()")

exit()

