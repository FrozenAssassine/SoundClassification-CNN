import tensorflow as tf
import matplotlib.pyplot as plt


class TrainingMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(TrainingMetricsCallback, self).__init__()
        self.epochs = []
        self.losses = []
        self.accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs.append(epoch + 1)
        self.losses.append(logs.get('loss', 0))
        self.accuracies.append(logs.get('accuracy', 0))
        self.val_losses.append(logs.get('val_loss', 0))
        self.val_accuracies.append(logs.get('val_accuracy', 0))

        print(f"\nEpoch {epoch + 1}:")
        print(f"  Loss: {logs.get('loss', 0):.4f}, Accuracy: {logs.get('accuracy', 0):.4f}")
        if 'val_loss' in logs:
            print(f"  Val Loss: {logs.get('val_loss', 0):.4f}, Val Accuracy: {logs.get('val_accuracy', 0):.4f}")

    def plot_metrics(self):
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(self.epochs, self.losses, 'b-', label='Training Loss')
        ax1.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss')

        ax1.set_title('Loss over Epochs')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.epochs, self.accuracies, 'b-', label='Training Accuracy')
        ax2.plot(self.epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Accuracy over Epochs')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()
