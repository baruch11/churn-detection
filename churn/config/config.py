
from calendar import c
from typing import Dict
import yaml

def read_yaml(path: str) -> dict:
    """Returns YAML file as dict"""
    with open(path, 'r') as file_in:
        config = yaml.unsafe_load(file_in)
    return config


    return 
def transform_to_object(path_of_models : str ,mapping_dict : Dict):
    """
    This function transform models name from string to class. 
    Ex : from "SVC()" to sklearn.SVC() """

    config = read_yaml("churn/config/config_template.yml")
    grid_parameters = config["model"][str(path_of_models)]
    model_list = list()
    for model in grid_parameters :
        model["pipeline__classifier"][0] = mapping_dict[model["pipeline__classifier"][0]]
        model_list.append(model)
    return model_list

