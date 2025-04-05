from model import create_model, predict, show_confusion_matrix
import tensorflow as tf

from data_processing import make_training_data
from model import create_compile_model
import config

if __name__ == "__main__":
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("model.keras", save_best_only=True)

    model = create_compile_model(learning_rate=0.01)

    train_dataset, test_dataset, test_x, test_y, name_index = make_training_data(
        config.DATA_CSV_PATH,
        config.AUDIO_DF_PATH,
        config.DATA_WAV_DIR,
        config.TRAINING_ITEM_COUNT,
        train_test_split=config.TRAIN_TEST_SPLIT,
        use_existing_df=config.USE_EXISTING_AUDIO_DATA
    )

    model.fit(train_dataset, epochs=config.EPOCHS, validation_data=test_dataset, callbacks=[checkpoint_cb])
    model.evaluate(test_dataset)

    show_confusion_matrix(model, name_index, test_x, test_y)
