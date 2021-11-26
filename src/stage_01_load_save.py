# This program will load data from AWS S3 bucket to local

import pandas as pd
import os
import shutil
from tqdm import tqdm
import logging
import argparse
from src.utils.all_utils import read_yaml, create_dir

stage_no = "One"

#logging
logging_str = "[%(asctime)s: %(levelname)s : %(module)s] : %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir, "runnning.log"), level= logging.INFO, format=logging_str, filemode="a")


# copy file from outside (AWS S3 bucket) to local dir
def copy_file(source_download_dir, local_data_dir):
    list_of_files = os.listdir(source_download_dir)
    N = len(list_of_files)

    for file in tqdm(list_of_files, colour= 'green', desc= f"Copying file from {source_download_dir} to {local_data_dir}"):
        src  = os.path.join(source_download_dir, file)
        dest = os.path.join(local_data_dir, file)
        shutil.copyfile(src, dest)
    logging.info(f"All files has been moved from {source_download_dir} to {local_data_dir}")


def get_data(config_path):
    config = read_yaml(config_path)
    print(f"source_download_dir is {config['source_download_dir']}")
    source_download_dirs = config['source_download_dir']
    local_data_dirs = config['local_data_dir']

    for source_download_dirs, local_data_dirs in tqdm(zip(source_download_dirs, local_data_dirs), colour='red', total=2, desc='List of the folders'):
        create_dir([local_data_dirs])
        copy_file(source_download_dirs, local_data_dirs)



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default= "config/configs.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info(f"/n>>>>>Stage {stage_no} has been started")
        get_data(config_path= parsed_args.config)
        logging.info(f"Stage {stage_no} has been completed")

    except Exception as e:
        logging.exception(e)
        raise e


