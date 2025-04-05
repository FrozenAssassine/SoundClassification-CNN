import os
import random
from data_processing import parse_data
from model import create_model, predict
import config

if __name__ == "__main__":

    _, _, name_index = parse_data(config.DATA_CSV_PATH, config.TRAINING_ITEM_COUNT)

    model = create_model()
    model.summary()

    model.load_weights(config.MODEL_PATH)

    # load random items from disk
    for file in random.choices(os.listdir(config.DATA_WAV_DIR), k=10):
        predicted_class, confidence = predict(model, os.path.join(config.DATA_WAV_DIR, file))
        print(f"Predicted {name_index[predicted_class]} with {confidence}")
