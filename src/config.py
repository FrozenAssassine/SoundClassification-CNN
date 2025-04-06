EPOCHS = 33
TRAINING_ITEM_COUNT = -1
TRAIN_TEST_SPLIT = 5000  # number of train items, rest is test
BATCH_SIZE = 8
LEARNING_RATE = 0.01

AUDIO_DF_PATH = "../out/audio_data.h5"
MODEL_PATH = "../out/model_new.keras"

TRAIN_NEW = True  # train completely new or use existing weights and train based on them
USE_EXISTING_AUDIO_DATA = True

DATA_CSV_PATH = "/mnt/f/NN DATASETS/BirdSounds/Archive1/bird_songs_metadata.csv"
DATA_WAV_DIR = "/mnt/f/NN DATASETS/BirdSounds/Archive1/wavfiles"
DATA_IMAGE_OUT_DIR = "/mnt/NN DATASETS/BirdSounds/Archive1/images"
