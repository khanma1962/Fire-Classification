# This stage will prepare for callbacks

import os
from tqdm import tqdm
import argparse
import logging

from src.utils.all_utils import create_dir, read_yaml
from src.utils.callbacks import create_and_save_tensorboard_callbacks, create_and_save_checkpoint_callbacks


stage_no = 'Three'

#logging
logging_str = "[%(asctime)s: %(levelname)s : %(module)s] : %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir, "runnning.log"), level= logging.INFO, format=logging_str, filemode="a")


def prepare_callbacks(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config['artifacts']
    artifacts_dir = artifacts['ARTIFACTS_DIR']
    tensorboard_log_dir = os.path.join(artifacts_dir, artifacts['TENSORBOARD_ROOT_LOG_DIR'])
    checkpoints_dir = os.path.join(artifacts_dir, artifacts['CHECKPOINTS_DIR'])
    callbacks_dir = os.path.join(artifacts_dir, artifacts['CALLBACKS_DIR'])

    create_dir([tensorboard_log_dir, checkpoints_dir, callbacks_dir ])

    create_and_save_tensorboard_callbacks(callbacks_dir, tensorboard_log_dir)
    create_and_save_checkpoint_callbacks(callbacks_dir, checkpoints_dir)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default= "config/configs.yaml")
    args.add_argument("--params", "-p", default= "params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info(f"\n>>>>>Stage {stage_no} has been started")
        prepare_callbacks(config_path= parsed_args.config, params_path=parsed_args.params)
        logging.info(f"Stage {stage_no} has been completed")

    except Exception as e:
        logging.exception(e)
        raise e

