# Importing all necessary libraries
import json
from google.protobuf import message
import numpy as np
import librosa
import matplotlib.pyplot as plt
import noisereduce as nr
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import IPython
import os
from tkinter import filedialog
from tkinter import *
from tkinter import messagebox

# Initialize main ui screen
main_ui = Tk()

# Load Segment Audio Classification model
model_path = r"Models/"
model_name = "2021_03_05_16_36_55"

# Model reconstruction from a json file
with open(model_path + model_name + '.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights(model_path + model_name + '.h5')

# Replicate LableEncoder
# LabelEncoder is used for normalizing label values
lb = LabelEncoder()

# using fit_transform
lb.fit_transform((['CanOpening', 'CarHorn', 'Cat', 'ChirpingBirds', 'Clapping', 'ClockAlarm', 'Cow', 'CracklingFire', 'Crow', 'CryingBaby', 'Dog', 'Door_or_WoodCreaks', 'Engine', 'Fireworks', 'GlassBreaking', 'HandSaw', 'Helicopter', 'Hen', 'Laughing', 'Night', 'Pig', 'Rain', 'Rooster', 'SeaWaves', 'Siren', 'Snoring', 'Thunderstorm', 'Train', 'TwoWheeler', 'VaccumCleaner', 'WaterDrops', 'Wind']))

# SPLIT INPUTED AUDIO
def split_audio(audio_data, w, h, threshold_level, tolerance=10):
    split_map = []
    start = 0
    data = np.abs(audio_data)  # To get the absolute value
    # np.mean is used to compute arithmetic mean of threshold level
    threshold = threshold_level*np.mean(data[:25000])
    inside_sound = False  # To ensure that there is no audio playback
    near = 0

    for i in range(0, len(data)-w, h):
        win_mean = np.mean(data[i:i+w])  # mean of window length
        # if sound is having higher threshold than average threshold of audio file and its playable
        if(win_mean > threshold and not (inside_sound)):
            inside_sound = True  # playable audio-clip
            start = i  # reset the audio starting point
        if(win_mean <= threshold and inside_sound and near > tolerance):
            inside_sound = False
            near = 0
            split_map.append([start, i])  # Insert trimmed audio file
        if(inside_sound and win_mean <= threshold):
            near += 1
    return split_map  # return trimmed audio file

# FUNCTION FOR NORMALIZATION
def minMaxNormalize(arr):
    mn = np.min(arr)  # returns element wise minimum of array elements.
    mx = np.max(arr)  # returns element wise maximum of array elements.
    return (arr-mn)/(mx-mn)

# FUNCTION TO PREDICT SOUND
def predictSound(X):
    
    # Returns magnitude of frequency bin f at frame t
    stfts = np.abs(librosa.stft(X, n_fft=512, hop_length=256, win_length=512))
    stfts = np.mean(stfts, axis=1)
    stfts = minMaxNormalize(stfts)
    result = model.predict(np.array([stfts]))  # Predict Output of model
    predictions = [np.argmax(y) for y in result]
    return lb.inverse_transform([predictions[0]])[0]

# FUNCTION TO RUN PREDICTION
def run_func():
# Import file using tkinter
    main_ui.filename = filedialog.askopenfilename(
       initialdir="/", filetypes=(("Audio files", "*.wav"), ("all files", "*.*")))

    if main_ui.filename.endswith('.wav'):

        raw_audio, sr = librosa.load(main_ui.filename)
        noisy_part = raw_audio[0:50000]

        nr_audio = nr.reduce_noise(
            audio_clip=raw_audio, noise_clip=noisy_part, verbose=False)

        clip, index = librosa.effects.trim(nr_audio, top_db=20, frame_length=512, hop_length=64)
        result = predictSound(clip)
        if(result == "CanOpening" or "CarHorn" or "Clapping" or "ClockAlarm" or "Cow" or "Crow" or "CryingBaby" or "Dog" or "Engine" or "Fireworks" or "GlassBreaking" or "HandSaw" or "Helicopter" or "Laughing" or "Siren" or "Snoring" or "Thunderstorm" or "Train" or "TwoWheeler" or "VaccumCleaner"):
            messagebox.showwarning("Result","It's sound of a "+result+" and it's noisy for children!")
        
        else:
            messagebox.showinfo("Children are in pretty much good environment!")
        
        del result

    else:
        messagebox.showinfo("Error","Wrong file selected/No file Selected")

# ALL FUNCTIONS END HERE

# MAIN FUNCTION - working starts from here
main_ui.configure(bg='pink')

main_ui.title("Environmental Audio Scene and Sound Event Recogntion")
main_ui.geometry('850x550')

entry_label = Label(text='\n\nWelcome Geek!\n\nIts a project presented by \n1.Shubham Thombare\n2.Vishal Kajale\n3.Ankush Soni\n4.Kunal Sonar\n\n\n Please select Your file to upload\n',
                    fg='purple', bg='pink', font=('Arial', 15, 'italic')).pack()
file_add_button = Button(text="Select File", fg="pink", bg="purple", command=run_func, font=('opensans', 12, 'bold'),border=0,width=20).place(x=220, y=350)
exit_button = Button(main_ui, text='Exit', fg="pink", bg='purple',
                     command=main_ui.destroy, font=('opensans', 12, 'bold'),border=0,width=20).place(x=470, y=350)
main_ui.mainloop()