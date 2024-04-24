import tkinter as tk
from tkinter import filedialog, Label, Button
import pygame
from tkinter.ttk import Progressbar
from tensorflow.keras.models import model_from_json
import numpy as np
import librosa

# Function to load the voice tone analysis model
def LoadVoiceToneModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Function to extract features from an audio file
def ExtractAudioFeatures(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Function to predict voice tone
def PredictVoiceTone(audio_file, model):
    feature = ExtractAudioFeatures(audio_file)
    prediction = model.predict(np.expand_dims(feature, axis=0))
    predicted_class = np.argmax(prediction)
    return predicted_class

# Function to handle audio file upload and prediction
def UploadAudioFile():
    global audio_file_path
    audio_file_path = filedialog.askopenfilename()
    play_button.config(state="normal")
    predict_button.config(state="normal")

def PlayAudio():
    pygame.init()
    pygame.mixer.music.load(audio_file_path)
    pygame.mixer.music.play()
    progress_bar.start(100)

# Function to predict voice tone and display result
def PredictVoiceToneAndDisplayResult():
    predicted_class = PredictVoiceTone(audio_file_path, voice_tone_model)
    label_result.config(text="Predicted Voice Tone: {}".format(emotions[predicted_class]))
    progress_bar.stop()

# GUI setup
top = tk.Tk()
top.geometry('800x600')
top.title('Voice Tone Analysis')
top.configure(background='#CDCDCD')

heading = Label(top, text='Voice Tone Analysis', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack(side='top')

label_result = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
label_result.pack(side='bottom', pady=20)

# Load the voice tone analysis model
voice_tone_model = LoadVoiceToneModel("model_aVTN.json", "model_weights.h5")

emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

upload_button = Button(top, text="Upload Audio File", command=UploadAudioFile, padx=10, pady=5)
upload_button.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload_button.pack(side='top', pady=50)

play_button = Button(top, text="Play Audio", command=PlayAudio, padx=10, pady=5, state="disabled")
play_button.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
play_button.pack(side='top', pady=20)

progress_bar = Progressbar(top, orient=tk.HORIZONTAL, length=100, mode='indeterminate')
progress_bar.pack(side='top', pady=5)

predict_button = Button(top, text="Predict Voice Tone", command=PredictVoiceToneAndDisplayResult, padx=10, pady=5, state="disabled")
predict_button.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
predict_button.pack(side='top', pady=20)

top.mainloop()
