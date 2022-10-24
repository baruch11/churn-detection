"""Compute training from a csv file"""

import argparse
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from churn.domain.churn_model import ChurnModelFinal
from churn.infrastructure.bank_customers import BankCustomersData


PARSER = argparse.ArgumentParser(
    description='Compute training from a csv file')
PARSER.add_argument('--indicators_csv', '-i', type=str,
                    help='indicators csv file name')
PARSER.add_argument('--customers_csv', '-c', type=str,
                    help='customers csv file name')


args = PARSER.parse_args()

#instanciation des données à partir de la classe BankCustomersData
#du module /infrastructure/bank_customers.py
bcd = BankCustomersData(args.indicators_csv, args.customers_csv)
raw_data = bcd.load_data()

X_train, X_test, y_train, y_test = train_test_split(raw_data.drop(columns=["CHURN"]), raw_data["CHURN"], test_size=0.20, random_state=33)

#training du modèle à partir de la classe DummyChurnModel
# du module /domain/churn_model.py
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


model.save()
