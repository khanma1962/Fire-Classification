
import os
import argparse
import logging
from src.utils.all_utils import read_yaml, create_dir
from src.utils.callbacks import get_callbacks
from src.utils.models import load_full_model , get_unique_filename_to_save_model
from src.utils.data_management import train_valid_generator


stage_no = 'Four'

#logging
logging_str = "[%(asctime)s: %(levelname)s : %(module)s] : %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir, "runnning.log"), level= logging.INFO, format=logging_str, filemode="a")


def train_model(config_path, params_path):

    config = read_yaml(config_path)
    params = read_yaml(params_path)
    artifacts = config['artifacts']
    artifacts_dir = artifacts['ARTIFACTS_DIR']

    train_model_dir_path = os.path.join(artifacts_dir, artifacts['TRAINED_MODEL_DIR'])
    create_dir([train_model_dir_path])

    untrained_full_model_path = os.path.join(artifacts_dir, artifacts['UPDATED_BASE_MODEL_DIR'], artifacts['UPDATED_BASE_MODEL_NAME'])
    model = load_full_model(untrained_full_model_path)

    callbacks_dir_path = os.path.join(artifacts_dir, artifacts['CALLBACKS_DIR'])
    callbacks = get_callbacks(callbacks_dir_path)

    train_generator, valid_generator = train_valid_generator(
                data_dir = artifacts['DATA_DIR'],
                IMAGE_SIZE = tuple(params['IMAGE_SIZE'][:-1]), # takinf only 224, 224 from [224, 224, 3]
                BATCH_SIZE = params['BATCH_SIZE'],
                do_data_augmentation = params['AUGMENTATION']
            )

    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = valid_generator.samples // valid_generator.batch_size


    model.fit(
                train_generator,
                validation_data = valid_generator,
                epochs = params['EPOCHS'],
                steps_per_epoch = steps_per_epoch,
                validation_steps = validation_steps,
                callbacks = callbacks
        )

    logging.info("Training completed")

    trained_model_dir = os.path.join(artifacts_dir, artifacts['TRAINED_MODEL_DIR'])
    create_dir([trained_model_dir])

    model_file_name = get_unique_filename_to_save_model(trained_model_dir)
    complete_model_path = os.path.join(trained_model_dir, model_file_name)

    model.save(complete_model_path)

    logging.info(f"Model has been saved at {complete_model_path}")



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default= "config/configs.yaml")
    args.add_argument("--params", "-p", default= "params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info(f"\n>>>>>Stage {stage_no} has been started")
        train_model(config_path= parsed_args.config, params_path=parsed_args.params)
        logging.info(f"Stage {stage_no} has been completed")

    except Exception as e:
        logging.exception(e)
        raise e

