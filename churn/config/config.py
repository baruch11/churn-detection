from typing import Dict
import yaml
import os
from sklearn.preprocessing import StandardScaler
from interpret.glassbox import ExplainableBoostingClassifier

def read_yaml(path: os.path) -> dict:
    """Returns YAML file as dict"""
    with open(path, 'r') as file_in:
        config = yaml.unsafe_load(file_in)
    return config


def transform_to_object(file_path : os.path, path_of_models : str ,mapping_dict : Dict):
    """
    This function transform models name from string to class. 
    Ex : from "SVC()" to sklearn.SVC() """

    config = read_yaml(file_path)
    if path_of_models is not None:
        grid_parameters = config[str(path_of_models)]
    else :
        grid_parameters = config
    model_list = list()
    for model in grid_parameters :
        print(f"Model pipeline classifier : {grid_parameters}")
        model["pipeline__classifier"][0] = mapping_dict[model["pipeline__classifier"][0]]
        model_list.append(model)
    return model_list

def save_best_params_to_yaml (path : str ,best_params : tuple,model_name : str):
    """This function save the result of a Bayesian search into a special file used by the train model. """
    model_to_save = dict()
    model_to_save["pipeline__classifier"] = [model_name]
    print(f"best_params {best_params}")
    for param in best_params:
        # Skiping the result of pipeline__classifier, because this object coming from Bayesian search hasn't the right format. 
        if param != "pipeline__classifier":
            model_to_save[param] = best_params[param]
    model_to_save = {
        "model_parameters" : [model_to_save]
    }
    with open(path, 'w') as file:
        yaml.dump(model_to_save, file)
    return True