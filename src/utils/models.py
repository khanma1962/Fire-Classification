
import tensorflow as tf
import os
from tensorflow.python.keras.backend import flatten
import logging



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


