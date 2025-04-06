
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import tensorflow as tf
from keras.layers import (Activation, BatchNormalization, Conv2D,
                          Dense, Dropout, Flatten, InputLayer,
                          MaxPooling2D, Reshape)
from keras.regularizers import l2
from data_processing import process_audio
from keras.models import Sequential


def create_model():
    model = Sequential(name="model")

    model.add(InputLayer(shape=(128, 130)))
    model.add(Reshape((128, 130, 1)))
    model.add(Conv2D(64, (8, 8), activation='relu', padding='same'))
    model.add(Conv2D(64, (6, 6), activation='relu', padding='same'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(5))  # number of classifications
    model.add(Activation('softmax'))

    return model


def predict(model, audio_path):
    image_data_mel = process_audio(audio_path)

    # add batch dimension => (1, 128, 130)
    image_data_mel = np.expand_dims(image_data_mel, axis=0)

    # apply L2 normalization like in training
    image_data_mel = tf.keras.utils.normalize(image_data_mel)

    predictions = model.predict(image_data_mel)

    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class]

    return predicted_class, confidence


def show_confusion_matrix(model, name_index, test_x, test_y):
    pred_y = model.predict(test_x).argmax(axis=1, keepdims=True)
    true_y = test_y.argmax(axis=1, keepdims=True)

    confusion_matrix = sklearn.metrics.confusion_matrix(true_y, pred_y)
    confusion_matrix_display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=name_index)
    _, ax = plt.subplots(1, figsize=(8, 8))
    confusion_matrix_display.plot(ax=ax, xticks_rotation=10)
    plt.show()
