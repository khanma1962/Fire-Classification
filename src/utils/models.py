
import tensorflow as tf
import os
from tensorflow.python.keras.backend import flatten
import logging

from src.utils.all_utils import get_timestamp



def get_VGG16_model(input_shape,model_path):
    model = tf.keras.applications.VGG16(
            input_shape= input_shape,
            weights = 'imagenet',
            include_top = False
        )
    model.save(model_path)
    logging.info(f"Model has been saved at {model_path}")

    return model

def prepare_model(model, CLASSES , freeze_all , freeze_till , learning_rate ):
    if freeze_all:
        for layer in model.layers:
            layer.trainable = False
    elif (freeze_till is not None) and (freeze_till > 0):
        for layer in model.layers[:-freeze_till]:
            layer.trainalble = False

    # add full connected layer
    flatten_in = tf.keras.layers.Flatten()(model.output)
    prediction = tf.keras.layers.Dense(
                units = CLASSES,
                activation = 'softmax',
                )(flatten_in)

    full_model = tf.keras.Model(
                inputs = model.inputs,
                outputs = prediction
                )

    full_model.compile(
                optimizer = tf.keras.optimizers.SGD(learning_rate= learning_rate),
                loss = tf.keras.losses.CategoricalCrossentropy(),
                metrics = ['accuracy']
                )

    logging.info("Custom model is compiled and ready to train")
    full_model.summary()

    return full_model


def load_full_model(untrained_full_model_path):
    model = tf.keras.models.load_model(untrained_full_model_path)
    logging.info(f"Loading untrained model from {untrained_full_model_path} is completed")

    return model


def get_unique_filename_to_save_model(trained_model_dir, model_name='model'):
    timestamp = get_timestamp(model_name)
    unique_model_name = f"{timestamp}_.h5"
    unique_model_path = os.path.join(trained_model_dir, unique_model_name )

    return unique_model_name





