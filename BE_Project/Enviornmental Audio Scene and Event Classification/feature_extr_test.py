import librosa #Python package manager for music and audio analysis
import os #Provides functions for interacting with operating system
import numpy as np #Numpy is a python library that provides a simple yet powerful data structure :the n-dimensional array.This is the foundation on which alomost all the power of pythons data science toolkit is built and learning Numpy is the first step on any python data scientist journey.
import noisereduce as nr #Noise reduction in python using spectral gating. Noisereduce optionally uses tensorflow as backend to speed up FFT and gausian convolution.
# A fast Fourier transform (FFT) is an algorithm that computes the discrete Fourier transform (DFT) of a sequence, or its inverse (IDFT). Fourier analysis converts a signal from its original domain (often time or space) to a representation in the frequency domain and vice versa.
#  a Gaussian filter modifies the input signal by convolution with a Gaussian function; this transformation is also known as the Weierstrass transform.

# Function to save STFT features of an audio file
def save_STFT(file,name,activity,subject):
    # Read audio data
    audio_data,sample_rate = librosa.load(file)#librosa.load returns two things. The sample rate which means how many samples are recorded in one second and a 2D array where the first axis represents the recorded samples of amplitudes in the audio, the second axis represents the number of channels in the audio.
    # Load an audio file as a floating point time series
    # In case of audio files, it's the passage or communication channel in which a sound signal is transported from the player source to the speaker. ... Especially in the case of surround sound, more audio Channels are needed to create the feeling of sound being – as the name suggests – all around the listener.
    
    #Noise reduction
    noisy_part = audio_data[0:25000] #Section of audio file i.e noise
    reduced_noise = nr.reduce_noise(audio_clip=audio_data,noise_clip = noisy_part, verbose=False)

    # Trimming - Trim leading and trailing silence from an audio signal.
    trimmed, index = librosa.effects.trim(reduced_noise, top_db = 20, frame_length = 512, hop_length = 64)
    # returns the trimmed signal and the interval of y corresponding to the non-silent region: y_trimmed = y[index[0]:index[1]] (for mono) or y_trimmed = y[:, index[0]:index[1]] (for stereo).
    # top_db is the threshold below which sound is considered as silent
    # frame_lenth = The number of samples per analysis frame
    # hop_length = The number of samples between analysis frames

    # Extract features
    stft = np.abs(librosa.stft(trimmed, n_fft = 512,hop_length = 256, win_length = 512))
    # np.abs = To get their absolute values, we call the numpy. abs() function and pass in that array as an argument. As a result NumPy returns a new array, with the absolute value of each number in the original array.
    # librosa.stft = Short-time Fourier transform (STFT).The STFT represents a signal in the time-frequency domain by computing discrete Fourier transforms (DFT) over short overlapping windows.This function returns a complex-valued matrix D such that
    # np.abs(D[f, t]) is the magnitude of frequency bin f at frame t, and
    # np.angle(D[f, t]) is the phase of frequency bin f at frame t.
    # n_fft = length of the windowed signal after padding with zeros. The number of rows in the STFT matrix D is (1 + n_fft/2). The default value, n_fft=2048 samples, corresponds to a physical duration of 93 milliseconds at a sample rate of 22050 Hz, i.e. the default sample rate in librosa. This value is well adapted for music signals. However, in speech processing, the recommended value is 512, corresponding to 23 milliseconds at a sample rate of 22050 Hz. In any case, we recommend setting n_fft to a power of two for optimizing the speed of the fast Fourier transform (FFT) algorithm.

    # Save features as a numpy array
    np.save("STFT_features/"+ subject + "_" + name[:-4] + "_" + activity + ".npy",stft)
    # name[:-4] is to delete extension of file from filename
# Execution starts here
activities = ['Airplane','Breathing','BrushingTeeths','CanOpening','CarHorn','Cat','ChirpingBirds','ChurchBells','Clapping','ClockAlarm','Coughing','Cow','CracklingFire','Crow','CryingBaby','Dog','Door_or_WoodCreaks','DoorKnock','Drinking','Engine','Fireworks','FlyingInsects','Footsteps','Frog','GlassBreaking','HandSaw','Helicopter','Hen','Laughing','Night','Pig','PouringWater','Rain','Rooster','SeaWaves','Sheep','Siren','Sneezing','Thunderstorm','Train','VaccumCleaner','WashingMachine','WaterDrops','Wind']

subjects = ['s01','s02','s03','s04','s05']

for activity in activities:
    for subject in subjects:
        innerDir = subject + "/" + activity
        for file in os.listdir("Dataset_audio/" + innerDir):
            if(file.endswith(".wav")):
                save_STFT("Dataset_audio/"+ innerDir + "/" + file,file,activity,subject)
                print(subject,activity,file)