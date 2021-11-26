
# This file cotains all important utilities

import numpy as np
import os
import yaml
import time
import json
import logging

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


# to get the time stamp from the file
def get_timestamp(tb_logs):
    time_stamp = time.asctime().replace(' ', '_').replace(':', '_')
    unique_name = f"{tb_logs}_at_{time_stamp}"

    return unique_name



