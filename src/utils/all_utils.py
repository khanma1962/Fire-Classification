
# This file cotains all important utilities

import logging
import os
import yaml


# to create a list of directories
def create_dir(dirs: list):
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Directory has been created at {dir_path}")

# to read yaml file
def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
        logging.info(f"yaml file at {path_to_yaml} has been successfully read")

    return content
