# Importing all necessary libraries
import json
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
model_name = "2021_03_04_18_45_34[6.829300403594971, 0.3680555522441864]"

# Model reconstruction from a json file
with open(model_path + model_name + '.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights(model_path + model_name + '.h5')

# Replicate LableEncoder
# LabelEncoder is used for normalizing label values
lb = LabelEncoder()

# using fit_transform
lb.fit_transform(['Airplane','CanOpening','CarHorn','Clapping','ClockAlarm','Crow','CryingBaby','Dog','Engine','Fireworks','GlassBreaking','HandSaw','Helicopter','Laughing','Siren','Thunderstorm','Train','VaccumCleaner','other'])

# SPLIT GIVEN LONG AUDIO FILE IN SILENT PARTS
# Accepts audio numpy array audio_data,window_length w,hop_length h,threshold_level,tolerance
# WINDOW LENGTH - no of frames in current window
# HOP LENGTH = No of samples between successive frames
# THRESHOLD LEVEL = The threshold control sets the level at which the compression effect is engaged. Only when a level passes above the threshold will it be compressed. If the threshold level is set at say -10 dB, only signal peaks that extend above that level will be compressed.minimum threshold here is slemce th
# Higher tolerance to prevent small silence parts from splitting the audio
# Returns array containing arrays of [start,end] points of resulting audio clips


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

# Normalization is used to minimize the redudancy from a relation or set of relations.It is also used to eliminate the undesirable characteristics like insertion,update and deletion anomalies. Normalization Divides the larger table in smaller table and links them using relationship.
# Normalization function
# We are passing numpy array in function


def minMaxNormalize(arr):
    mn = np.min(arr)  # returns element wise minimum of array elements.
    mx = np.max(arr)  # returns element wise maximum of array elements.
    return (arr-mn)/(mx-mn)


def predictSound(X):
    # librosa.stft returns Short-time Fourier transform (STFT).
    # The STFT represents a signal in the time-frequency domain by computing discrete Fourier transforms (DFT) over short overlapping windows.
    # This function returns a complex-valued matrix D such that
    # np.abs(D[f, t]) is the magnitude of frequency bin f at frame t,

    # Returns magnitude of frequency bin f at frame t
    stfts = np.abs(librosa.stft(X, n_fft=512, hop_length=256, win_length=512))
    stfts = np.mean(stfts, axis=1)
    stfts = minMaxNormalize(stfts)
    result = model.predict(np.array([stfts]))  # Predict Output of model
    predictions = [np.argmax(y) for y in result]
    # np.argmax Returns the indices of the maximum values along an axis.
    # Transform binary labels back to multi-class labels.
    return lb.inverse_transform([predictions[0]])[0]

# Function to run prediction
def run_func():
# Import file using tkinter
    main_ui.filename = filedialog.askopenfilename(
       initialdir="/", filetypes=(("Audio files", "*.wav"), ("all files", "*.*")))

    if main_ui.filename.endswith('.wav'):

        raw_audio, sr = librosa.load(main_ui.filename)
        # Empherically selected noisy part position for every sample
        noisy_part = raw_audio[0:50000]

        # PERFORM NOISE REDUCTION
        # Noise Reduction Algorithm
        # Steps of algorithm
        # An FFT is calculated over the noise audio clip
        # Statistics are calculated over FFT of the the noise (in frequency)
        # A threshold is calculated based upon the statistics of the noise (and the desired sensitivity of the algorithm)
        # An FFT is calculated over the signal
        # A mask is determined by comparing the signal FFT to the threshold
        # The mask is smoothed with a filter over frequency and time
        # The mask is appled to the FFT of the signal, and is inverted

        nr_audio = nr.reduce_noise(
            audio_clip=raw_audio, noise_clip=noisy_part, verbose=False)

        clip, index = librosa.effects.trim(
            nr_audio, top_db=20, frame_length=512, hop_length=64)
            # trim is used to trim leading and trailing silence from audio
            # returns a trimmed signal and the interval of y corresponding to the non silent region.
        result = predictSound(clip)
        if(result == 'other' ):
            messagebox.showinfo("Result","Children are in safe environment")
        # print(result)
        else:
            messagebox.showinfo("Result","The sound is of "+ result + "Children are not in safe Environment")

    else:
        messagebox.showinfo("Error","Wrong file selected/No file Selected")

# MAIN FUNCTION - working starts from here
main_ui.configure(bg='pink')

main_ui.title("Environmental Audio Scene and Sound Event Recogntion")
main_ui.geometry('850x550')

entry_label = Label(text='\n\nWelcome Geek!\n\nIts a project presented by \n1.Shubham Thombare\n2.Vishal Kajale\n3.Ankush Soni\n4.Kunal Sonar\n\n\n Please select Your file to upload\n',
                    fg='purple', bg='pink', font=('Arial', 15, 'italic')).pack()
# messagebox.showinfo("Please select Audio file from your computer:")
file_add_button = Button(text="Select File", fg="pink", bg="purple", command=run_func, font=('opensans', 12, 'bold'),border=0,width=20).place(x=220, y=350)
exit_button = Button(main_ui, text='Exit', fg="pink", bg='purple',
                     command=main_ui.destroy, font=('opensans', 12, 'bold'),border=0,width=20).place(x=470, y=350)
main_ui.mainloop()