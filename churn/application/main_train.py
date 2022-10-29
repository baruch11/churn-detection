"""Compute training from a csv file"""

import argparse
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from churn.domain.churn_model import ChurnModelFinal
from churn.domain.domain_utils import get_train_test_split
from churn.config.config import retrieve_optimal_parameters



PARSER = argparse.ArgumentParser(
    description='Compute training from a csv file')

args = PARSER.parse_args()

X_train, X_test, y_train, y_test = get_train_test_split()

#Retrieve best_params dict of the chosen model
best_params = retrieve_optimal_parameters()[1]

#Model training based on optimal hyperparameters obtained in main_optimize
model = ChurnModelFinal()
model.pipe.set_params(**best_params)
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

