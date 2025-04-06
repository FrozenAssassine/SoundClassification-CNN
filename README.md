# 🔊 Sound Classification

Simple sound classification in Python using a Convolutional Neural Network (CNN).


## 🚀 How it works

Audio files are converted into NumPy arrays using **Mel-spectrogram conversion**.  
This process turns the raw audio data into a spectrogram image, capturing frequency vs time.  
The spectrogram is then used as input for a **CNN model**, which learns to classify different types of sounds.


## 🧰 Features

- Convert audio files into spectrograms using the MEL scale
- Feed image-like data into a CNN for training and prediction
- Easily customizable for different sound classification tasks


## 🛠 Requirements

- Python 3.8+
- TensorFlow / Keras
- Librosa
- NumPy
- Matplotlib (optional for spectrogram visualization)

##  Running
To train the model, use the train.py script: 
```bash
python train.py
``` 

To use the model use the test.py script:
```bash
python test.py
``` 
