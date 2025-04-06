from model import create_model, predict, show_confusion_matrix
import tensorflow as tf

from data_processing import make_training_data
import config
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
import matplotlib.pyplot as plt

from training_metrics_callback import TrainingMetricsCallback

if __name__ == "__main__":
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(config.MODEL_PATH, save_best_only=True)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=0.96)

    model = create_model()
    model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(learning_rate=lr_schedule),  metrics=['accuracy'])

    train_dataset, test_dataset, test_x, test_y, name_index = make_training_data(
        config.DATA_CSV_PATH,
        config.AUDIO_DF_PATH,
        config.DATA_WAV_DIR,
        config.TRAINING_ITEM_COUNT,
        train_test_split=config.TRAIN_TEST_SPLIT,
        use_existing_df=config.USE_EXISTING_AUDIO_DATA
    )

    # load existing weights and train based on them
    if not config.TRAIN_NEW:
        model.load_weights(config.MODEL_PATH)

    history = model.fit(train_dataset, epochs=config.EPOCHS, validation_data=test_dataset, callbacks=[checkpoint_cb])

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    model.evaluate(test_dataset)

    show_confusion_matrix(model, name_index, test_x, test_y)
