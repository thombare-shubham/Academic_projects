import librosa
import noisereduce as nr
from tkinter import filedialog

# import files
from plot import *

# FUNCTION TO DISPLAY NOISEREDUCTION
def noise_reduction():
    file = filedialog.askopenfilename(
       initialdir="/", filetypes=(("Audio files", "*.wav"), ("all files", "*.*")))

    audio_data,sample_rate = librosa.load(file)
    plotaudio(audio_data,"Audio file before Noise Cancellation")

    noisy_part = audio_data[0:25000]
    reduced_noise = nr.reduce_noise(audio_clip = audio_data,noise_clip=noisy_part,verbose = False)
    plotaudio(reduced_noise,"Audio file after noise cancellation")

    trimmed,index = librosa.effects.trim(reduced_noise,top_db=20,frame_length=512,hop_length=64)
    plotaudio(trimmed,"Audio file after trimming")