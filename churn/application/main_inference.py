"""Compute inferences from a csv file"""
import os
import sys

import argparse
import logging
import yaml


from churn.infrastructure.bank_customers import BankCustomersData
from churn.domain.churn_model import DummyChurnModel

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


PARSER = argparse.ArgumentParser(
    description='Compute inferences from a csv file')
PARSER.add_argument('--params', '-p', type=str,
                    default=os.path.join(ROOTDIR, "churn/config/config_template.yml"),
                    help="path to the parameters yaml file")
PARSER.add_argument('--debug', '-d', action='store_true',
                    help='activate debug logs')

args = PARSER.parse_args()

logging.getLogger().setLevel(logging.DEBUG)
if args.debug:
    logging.getLogger().setLevel(logging.DEBUG)

logging.info("Reading parameters from %s", args.params)
with open(args.params, "r", encoding="utf-8") as fparam:
    cfg_dict = yaml.load(fparam, Loader=yaml.FullLoader)
    logging.debug("Parameters: %s", cfg_dict)

indicators_csv = os.path.join(ROOTDIR, cfg_dict["data"]["indicators_dataset"])
customers_csv = os.path.join(ROOTDIR, cfg_dict["data"]["customers_dataset"])

logging.info("Compute inferences for input files %s and %s",
             indicators_csv, customers_csv)

bcd = BankCustomersData(indicators_csv, customers_csv)
raw_data = bcd.load_data()

model = DummyChurnModel.load()
predictions = model.predict(raw_data)

output_csv = cfg_dict["inferences_output_file"]
predictions.to_csv(output_csv)

logging.info("Predictions written in %s", output_csv)
