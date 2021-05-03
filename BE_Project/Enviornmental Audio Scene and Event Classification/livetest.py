import numpy as np
import librosa
import matplotlib.pyplot as plt
import noisereduce as nr
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import IPython
import os

#Load segment audio classification model

model_path = r"Models/"
model_name = "audio_NN_New2021_02_11_16_07_49_acc_64"

# Model reconstruction from JSON file
with open(model_path + model_name + '.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights(model_path + model_name + '.h5')

# Replicate label encoder
lb = LabelEncoder()
lb.fit_transform(['DailyActivities', 'UnhealthyActivities', 'EnvironmentalActivities'])

#Some Utils

# Plot audio with zoomed in y axis
def plotAudio(output):
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,10))
    plt.plot(output, color='blue')
    ax.set_xlim((0, len(output)))
    ax.margins(2, -0.1)
    plt.show()

# Plot audio
def plotAudio2(output):
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4))
    plt.plot(output, color='blue')
    ax.set_xlim((0, len(output)))
    plt.show()

# Split a given long audio file on silent parts.
# Accepts audio numpy array audio_data, window length w and hop length h, threshold_level, tolerence
# threshold_level: Silence threshold
# Higher tolence to prevent small silence parts from splitting the audio.
# Returns array containing arrays of [start, end] points of resulting audio clips.
def split_audio(audio_data, w, h, threshold_level, tolerence=10):
    split_map = []
    start = 0
    data = np.abs(audio_data)
    threshold = threshold_level*np.mean(data[:25000])
    inside_sound = False
    near = 0
    for i in range(0,len(data)-w, h):
        win_mean = np.mean(data[i:i+w])
        if(win_mean>threshold and not(inside_sound)):
            inside_sound = True
            start = i
        if(win_mean<=threshold and inside_sound and near>tolerence):
            inside_sound = False
            near = 0
            split_map.append([start, i])
        if(inside_sound and win_mean<=threshold):
            near += 1
    return split_map

def minMaxNormalize(arr):
    mn = np.min(arr)
    mx = np.max(arr)
    return (arr-mn)/(mx-mn)

def predictSound(X):
    stfts = np.abs(librosa.stft(X, n_fft=512, hop_length=256, win_length=512))
    stfts = np.mean(stfts,axis=1)
    stfts = minMaxNormalize(stfts)
    result = model.predict(np.array([stfts]))
    predictions = [np.argmax(y) for y in result]
    return lb.inverse_transform([predictions[0]])[0]

# place concatenating audio files in a folder

folder = r"sound_clips/"
raw_audio, sr = librosa.load(folder + os.listdir(folder)[0])
for file in os.listdir(folder):
    data, rate = librosa.load(folder + file)
    raw_audio = np.concatenate((raw_audio,data))

noisy_part = raw_audio[0:50000]  # Empherically selected noisy_part position for every sample
nr_audio = nr.reduce_noise(audio_clip=raw_audio, noise_clip=noisy_part, verbose=False)

plotAudio(nr_audio)

sound_clips = split_audio(nr_audio, 10000, 2500, 15)

for intvl in sound_clips:
    clip, index = librosa.effects.trim(nr_audio[intvl[0]:intvl[1]], top_db=20, frame_length=512, hop_length=64) # Empherically select top_db for every sample
    print(predictSound(clip))
    plotAudio2(clip)
    IPython.display.display(IPython.display.Audio(data=clip, rate=sr))

