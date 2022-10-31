"""Compute training from a csv file"""

import os
import logging
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from churn.domain.churn_model import ChurnModelFinal , ChurnModelSelection
from sklearn.pipeline import Pipeline
from churn.domain.domain_utils import get_train_test_split
from churn.config.config import retrieve_optimal_parameters, transform_to_object
from churn.domain.bank_customers_dataset import FeaturesDataset

logging.getLogger().setLevel(logging.INFO)
PARSER = argparse.ArgumentParser(
    description='Compute training from a csv file')
PARSER.add_argument('--debug', '-d', action='store_true',
                    help='activate debug logs')

args = PARSER.parse_args()

if args.debug:
    logging.getLogger().setLevel(logging.DEBUG)

X_train, X_test, y_train, y_test = get_train_test_split()
latest_best_params = transform_to_object("churn/config/latest_model.yml","model_parameters")[0]
model_name = latest_best_params["pipe__classifier"][0]
#Removing model name for the set_params command
del latest_best_params["pipe__classifier"]
logging.debug("best params:\n%s", latest_best_params)

#Model training based on optimal hyperparameters obtained in main_optimize
model = ChurnModelSelection(pipe=Pipeline([('features', FeaturesDataset()),('scaler', StandardScaler()),('classifier', model_name)]))

model.set_params(**latest_best_params)

model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

#Accuracy score
print("Accuracy on test set", accuracy_score(y_test, y_pred_test))
print("Accuracy on train set", accuracy_score(y_train, y_pred_train))

#F1 score
print("F1 score on test set", f1_score(y_test, y_pred_test))
print("F1 score on train set", f1_score(y_train, y_pred_train))


print(f"score details on test set:\n{model.score_details(X_test, y_test)}")

model.save()

