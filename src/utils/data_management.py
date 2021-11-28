
import tensorflow as tf
import logging


def train_valid_generator(data_dir, IMAGE_SIZE , BATCH_SIZE , do_data_augmentation):
    
    datagenerator_kwarg = dict(
            rescale = 1.0/255.,
            validation_split = 0.2
        )

    dataflow_kwarg = dict(
            target_size = IMAGE_SIZE,
            batch_size = BATCH_SIZE,
            interpolation = "bilinear"
        )

    valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwarg)

    valid_generator = valid_datagenerator.flow_from_directory(
            directory = data_dir,
            subset = "validation",
            shuffle = False,
            **dataflow_kwarg
        )
    
    if do_data_augmentation:
        train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range = 40,
            horizontal_flip = True,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            shear_range = 0.2,
            zoom_range = 0.2,
            **datagenerator_kwarg
        )
        logging.info("Augumentation is used for training")
    else:
        train_datagenerator = valid_datagenerator
        logging.info(f"Augumentation is NOT used here")


    train_generator = train_datagenerator.flow_from_directory(
            directory = data_dir,
            subset = "training",
            shuffle = False,
            **dataflow_kwarg
        )

    logging.info(f"Train and Validation generators are created")

    return train_generator, valid_generator
