# This program will download the model and prepare for the training

import os
import argparse
import logging
import io
from tqdm import tqdm
from src.utils.all_utils import read_yaml, create_dir
from src.utils.models import get_VGG16_model, prepare_model



stage_no = 'Two'

#logging
logging_str = "[%(asctime)s: %(levelname)s : %(module)s] : %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir, "runnning.log"), level= logging.INFO, format=logging_str, filemode="a")


def prepare_base_model(config_path, param_path):
    config = read_yaml(config_path)
    params = read_yaml(param_path)

    artifacts = config['artifacts']
    artifacts_dir = artifacts['ARTIFACTS_DIR']

    base_model_dir = artifacts['BASE_MODEL_DIR']
    base_model_name = artifacts['BASE_MODEL_NAME']
    base_model_dir_path = os.path.join(artifacts_dir, base_model_dir)

    create_dir([base_model_dir_path])

    base_model_path = os.path.join(base_model_dir_path, base_model_name)
    
    model = get_VGG16_model(
            input_shape = params['IMAGE_SIZE'],
            model_path = base_model_path
            )

    full_model = prepare_model(model,
            CLASSES = params['CLASSES'],
            freeze_all = True,
            freeze_till = None,
            learning_rate = 0.001 
            )

    updated_base_model_path = os.path.join(
            base_model_dir_path, 
            artifacts['UPDATED_BASE_MODEL_NAME']
            )
    
    logging.info(f"{full_model.summary()}")


    full_model.save(updated_base_model_path)



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default= "config/configs.yaml")
    args.add_argument("--params", "-p", default= "params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info(f"/n>>>>>Stage {stage_no} has been started")
        prepare_base_model(config_path= parsed_args.config, param_path=parsed_args.params)
        logging.info(f"/n>>>>>Stage {stage_no} has been completed")

    except Exception as e:
        logging.exception(e)
        raise e

