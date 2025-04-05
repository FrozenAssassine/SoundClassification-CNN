import numpy as np
import random
import librosa
import os
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from keras.utils import to_categorical
import tensorflow as tf
import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, Reshape, InputLayer, TimeDistributed, Conv1D, MaxPooling1D, LSTM
from keras.models import Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
import sklearn.metrics

DATA_CSV_PATH = "/mnt/f/NN DATASETS/BirdSounds/Archive1/bird_songs_metadata.csv"
DATA_WAV_DIR = "/mnt/f/NN DATASETS/BirdSounds/Archive1/wavfiles"
DATA_IMAGE_OUT_DIR = "/mnt/NN DATASETS/BirdSounds/Archive1/images"
TRAIN = False
NUMBER_OF_ENTRYS = -1

name_index = []  # index of string in the array reprersents the classification index


def process_audio(audio_file):
    y, sr = librosa.load(audio_file, duration=10)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec


def parse_data(row_count):
    csv = pd.read_csv(DATA_CSV_PATH)
    csv["name"] = csv["name"].str.strip()

    species = csv["name"].unique()

    X = []
    y = []

    for index, bird in enumerate(species):
        name_index.append(bird)
        bird_data = csv[csv["name"] == bird].iloc[:row_count]
        for _, row in bird_data.iterrows():
            X.append(row["filename"])
            y.append(index)

    return X, y  # X => audio file, y = classification index, name_index = index vs class name = {"index": 0, "class": "Class_XY"}


def make_data(row_count):
    audio_files, indexes = parse_data(row_count)

    progress = 0
    mel_spectograms = []
    for file in audio_files:
        mel_spectograms += [process_audio(os.path.join(DATA_WAV_DIR, file))]
        progress += 1
        if progress % 100 == 0:
            print(f"MEL Creation Progress => Class {progress}/{len(audio_files)}")

    return mel_spectograms, indexes


def create_shuffled_dataframe(row_count):
    specs, indexes = make_data(row_count)

    df = pd.DataFrame({'mel_spec': specs, 'class': indexes})
    shuffled_df = df.sample(frac=1).reset_index(drop=True)

    return shuffled_df


def create_model():
    model = keras.models.Sequential(name="model")

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


def make_training_data(row_count, train_test_split, use_existing_df=False):
    # load existing dataframe from file or create new one => prompts to save again
    if use_existing_df:
        train_df = pd.read_pickle("audio_data.h5")
    else:
        train_df = create_shuffled_dataframe(row_count)
        if input("Save dataframe? [yes/no]") == "yes":
            train_df.to_pickle("audio_data.h5")
            print("Saved dataframe to audio_data.h5")

    (train_x, train_y) = train_df["mel_spec"][0:train_test_split].values, train_df["class"][0:train_test_split].values
    (test_x, test_y) = train_df["mel_spec"][train_test_split:-1].values, train_df["class"][train_test_split:-1].values

    unique_class_count = len(train_df["class"].unique())

    # one hot encode class the class names
    test_y = to_categorical(test_y, num_classes=unique_class_count)
    train_y = to_categorical(train_y, num_classes=unique_class_count)

    train_x = np.stack(train_x[:])
    test_x = np.stack(test_x[:])

    # L2 normalization: data from: [100, 50, -100] to [0.6667, 0.3333, -0.6667]
    train_x = tf.keras.utils.normalize(train_x)
    test_x = tf.keras.utils.normalize(test_x)

    # create the dataset for training, nparray to tensorflow training data:
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))

    # create batches from the dataset:
    train_dataset = train_dataset.batch(8)
    test_dataset = test_dataset.batch(8)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset, test_x, test_y


def predict(audio_path):
    image_data_mel = process_audio(audio_path)

    # add batch dimension => (1, 128, 130)
    image_data_mel = np.expand_dims(image_data_mel, axis=0)

    # apply L2 normalization like in training
    image_data_mel = tf.keras.utils.normalize(image_data_mel)

    predictions = model.predict(image_data_mel)

    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class]

    return predicted_class, confidence


def show_confusion_matrix(test_x, test_y):
    pred_y = model.predict(test_x).argmax(axis=1, keepdims=True)
    true_y = test_y.argmax(axis=1, keepdims=True)

    confusion_matrix = sklearn.metrics.confusion_matrix(true_y, pred_y)
    confusion_matrix_display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=name_index)
    fig, ax = plt.subplots(1, figsize=(8, 8))
    confusion_matrix_display.plot(ax=ax, xticks_rotation=10)
    plt.show()


checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("model.keras", save_best_only=True)

model = create_model()
model.summary()

parse_data(NUMBER_OF_ENTRYS)

if TRAIN:
    train_dataset, test_dataset, test_x, test_y = make_training_data(-1, 5000, True)
    model.fit(train_dataset, epochs=55, validation_data=test_dataset, callbacks=[checkpoint_cb])

    model.evaluate(test_dataset)

    # show_confusion_matrix(test_x, test_y)

model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])  # lr=0.01 decay=1e-6)

model.load_weights("model.keras")

for file in random.choices(os.listdir(DATA_WAV_DIR), k=10):
    predicted_class, confidence = predict(os.path.join(DATA_WAV_DIR, file))
    print(f"Predicted {name_index[predicted_class]} with {confidence}")
