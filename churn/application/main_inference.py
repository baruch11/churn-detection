"""Compute inferences from a csv file"""
import argparse
import logging

from churn.infrastructure.bank_customers import BankCustomersData
from churn.domain.churn_model import DummyChurnModel

PARSER = argparse.ArgumentParser(
    description='Compute inferences from a csv file')
PARSER.add_argument('--indicators_csv', '-i', type=str,
                    help='indicators csv file name')
PARSER.add_argument('--customers_csv', '-c', type=str,
                    help='customers csv file name')
PARSER.add_argument('--output_csv', '-o', type=str,
                    help='output predictions csv file name')

args = PARSER.parse_args()

logging.info("Compute inferences for input files %s and %s",
             args.indicators_csv, args.customers_csv)

bcd = BankCustomersData(args.indicators_csv, args.customers_csv)
raw_data = bcd.load_data()


model = DummyChurnModel.load()
predictions = model.predict(raw_data)

predictions.to_csv(args.output_csv)

logging.info("Predictions written in %s", args.output_csv)
