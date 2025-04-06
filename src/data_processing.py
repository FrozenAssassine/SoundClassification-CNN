import librosa
import os
import pandas as pd
import numpy as np

from keras.utils import to_categorical
import tensorflow as tf
import config


def process_audio(audio_file):
    y, sr = librosa.load(audio_file, duration=10)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec


def parse_data(csv_path, row_count):
    csv = pd.read_csv(csv_path)
    csv["name"] = csv["name"].str.strip()

    species = csv["name"].unique()

    X = []
    y = []
    name_index = []
    for index, bird in enumerate(species):
        name_index.append(bird)
        bird_data = csv[csv["name"] == bird].iloc[:row_count]
        for _, row in bird_data.iterrows():
            X.append(row["filename"])
            y.append(index)

    return X, y, name_index  # X => audio file, y = classification index


def make_data(csv_path, data_wav_dir, row_count):
    audio_files, indexes, name_index = parse_data(csv_path, row_count)

    progress = 0
    mel_spectograms = []
    for file in audio_files:
        mel_spectograms += [process_audio(os.path.join(data_wav_dir, file))]
        progress += 1
        if progress % 100 == 0:
            print(f"MEL Creation Progress => Class {progress}/{len(audio_files)}")

    return mel_spectograms, indexes, name_index


def create_shuffled_dataframe(csv_path, data_wav_dir, row_count):
    specs, indexes, name_index = make_data(csv_path, data_wav_dir, row_count)

    df = pd.DataFrame({'mel_spec': specs, 'class': indexes})
    shuffled_df = df.sample(frac=1).reset_index(drop=True)

    return shuffled_df, name_index


def load_dataframe_data(data_df_path, row_count, csv_path, data_wav_dir, use_existing_df):
    # load existing dataframe from file or create new one => prompts to save again
    if use_existing_df:
        train_df = pd.read_pickle(data_df_path)
        train_df = train_df.sample(frac=1).reset_index(drop=True)

        _, _, name_index = parse_data(csv_path, row_count)
    else:
        train_df, name_index = create_shuffled_dataframe(csv_path, data_wav_dir, row_count)
        if input("Save dataframe? [yes/no]") == "yes":
            train_df.to_pickle(data_df_path)
            print(f"Saved dataframe to {data_df_path}")

    return train_df, name_index


def make_training_data(csv_path, data_df_path, data_wav_dir, row_count, train_test_split, use_existing_df=False):

    train_df, name_index = load_dataframe_data(data_df_path, row_count, csv_path, data_wav_dir, use_existing_df)

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
    train_dataset = train_dataset.batch(config.BATCH_SIZE)
    test_dataset = test_dataset.batch(config.BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset, test_x, test_y, name_index
