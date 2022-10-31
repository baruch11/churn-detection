"""Compute training from a csv file"""

import sys
import logging
import argparse


from sklearn.metrics import accuracy_score, f1_score
from churn.domain.churn_model import ChurnModelFinal
from churn.domain.domain_utils import get_train_test_split

logging.getLogger().setLevel(logging.INFO)
PARSER = argparse.ArgumentParser(
    description='Compute training from a csv file')
PARSER.add_argument('--debug', '-d', action='store_true',
                    help='activate debug logs')

args = PARSER.parse_args()

if args.debug:
    logging.getLogger().setLevel(logging.DEBUG)


try:
    X_train, X_test, y_train, y_test = get_train_test_split()
except FileNotFoundError as emsg:
    print(emsg)
    logging.error("%s", emsg)
    logging.info("Import and copy the train dataset (1_-_indicators.csv and "
                 "1_-_customers.csv) to churn/data/")
    sys.exit(1)


X_train, X_test, y_train, y_test = get_train_test_split()

#Model training based on optimal hyperparameters obtained in main_optimize
model = ChurnModelFinal()
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

