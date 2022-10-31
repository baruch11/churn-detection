# Churn Modelling - Salima Charles Emeric

This project is churn prediction application. 
The model detects churn from customers information such as 'BALANCE', 'NB_PRODUITS', 'CARTE_CREDIT', 'SALAIRE', 'SCORE_CREDIT', 'PAYS', 'SEXE', 'AGE', 'MEMBRE_ACTIF'


## Installation

Clone the project and create a virtual environment with all the dependencies.

	git clone git@gitlab.com:yotta-academy/mle-bootcamp/projects/ml-project/project-1-fall-2022/churn-modelling-salima-charles-emeric.git
    cd churn-modelling-salima-charles-emeric/ 
    python3 -m venv .venv  
    source .venv/bin/activate
    pip install -r requirements.txt

## Make inferences

Create a yaml file <inference_file>.yml indicating the path of the 2 files on which you wish to make churn predictions and the output file.
The 2 input files are csv files on the same format of the 2 examples:  _\<rootdir\>/data/inference_test_customers.csv_ and _\<rootdir\>/data/inference_test_indicators.csv_
See an example in <rootdir>/churn/config/inference-template.yml

    cd <rootdir> # root of the project
    source activate.sh
    python churn/application/main_inference.py --params <relative path to inference file from rootdir>

The output file is a csv file (by default in _<rootdir>/data/predictions.csv_) with 2 columns, "ID_CLIENT", "CHURN_PREDICTION"" 

## Train the model 

The model in the repository (<rootdir>/data/model/ChurnModelFinal.pkl) is already trained.   
If you wish to train it with a new dataset (2 csv files containing customers and indicators), adapt _data_ field in <rootdir>/churn/config/config_template.yml and make the following steps:  

	cd rootdir
	source activate.sh
	python3 churn/application/main_train.py


## Bayesian Optimization process

The model and its hyper-parameters have been chosen with the main_optimize.py script.

All informations with 
<code>python3 churn/application/main_optimize.py -h</code>
Options are : -models, -GridSearchModel, -n_iter.

#### Find the best model :
<code>python3 churn/application/main_optimize.py -m</code>
Return scores of each of the model with default params

#### Find best params for a specific model :
<code>python3 churn/application/main_optimize.py -g "SVC() -n 10"</code>
Return score and best params performing Bayes optimization for the trained and test model.
Save the config automatically. 


## Unit tests

	pytest
