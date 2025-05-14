import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

# Constants
SAMPLE_RATE = 22050
DURATION = 3
FILENAME = "recorded_audio.wav"
MODEL_PATH = "emotion_recognition_model.h5"
ENCODER_PATH = "encoder.joblib"

# Label names (must match training order)
emotion_labels = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'pleasant surprise', 'sadness']

# Load model and encoder
model = load_model(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

# Extract MFCC features
def extract_mfcc(filename, duration=3, offset=0.5):
    y, sr = librosa.load(filename, duration=duration, offset=offset)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Predict emotion
def predict_emotion(filename):
    mfcc = extract_mfcc(filename)
    mfcc = np.expand_dims(mfcc, axis=-1)  # Shape: (40, 1)
    mfcc = np.expand_dims(mfcc, axis=0)   # Shape: (1, 40, 1)
    prediction = model.predict(mfcc)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_one_hot = np.zeros_like(prediction)
    predicted_one_hot[0, predicted_class] = 1
    emotion = encoder.inverse_transform(predicted_one_hot)
    return emotion[0], prediction[0]

# Record and Predict Handler
def record_and_predict():
    try:
        status_label.config(text="Recording...")
        root.update()
        recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()
        write(FILENAME, SAMPLE_RATE, recording)

        status_label.config(text="Predicting Emotion...")
        root.update()

        emotion, probabilities = predict_emotion(FILENAME)
        result_label.config(text=f"Predicted Emotion: {emotion}", fg="blue")

        details = "\n".join([f"{label}: {prob:.4f}" for label, prob in zip(emotion_labels, probabilities)])
        probabilities_text.delete('1.0', tk.END)
        probabilities_text.insert(tk.END, details)

        status_label.config(text="Done.")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI Setup
root = tk.Tk()
root.title("🎤 Speech Emotion Recognizer")
root.geometry("420x500")
root.resizable(False, False)

title = tk.Label(root, text="Speech Emotion Detection", font=("Helvetica", 16, "bold"))
title.pack(pady=20)

record_btn = tk.Button(root, text="🎙️ Record & Predict", font=("Helvetica", 14), command=record_and_predict)
record_btn.pack(pady=10)

result_label = tk.Label(root, text="", font=("Helvetica", 14))
result_label.pack(pady=10)

status_label = tk.Label(root, text="", font=("Helvetica", 10), fg="green")
status_label.pack()

probabilities_text = tk.Text(root, height=10, width=48, font=("Courier", 10))
probabilities_text.pack(pady=10)

root.mainloop()
