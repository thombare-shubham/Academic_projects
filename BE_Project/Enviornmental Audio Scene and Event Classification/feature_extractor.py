import librosa #Python package manager for music an audio analysis
import os #Provides functions for interacting with operating system
import numpy as np #Numpy is a python library that provides a simple yet powerful data structure:the n-dimensional array. This is the foundation on which almost all power of python's data science toolkit is built and learning NumPy is the first step on any python data scientist journey.
import noisereduce as nr #Noise reduction in python using spectral gating.Noisereduce optionally uses tensorflow as a backend to speed up FFT and gaussian convolution.


def save_STFT(file, name, activity, subject):
    # read audio data
    audio_data, sample_rate = librosa.load(file) #librosa.load returns two things. The sample rate which means how many samples are recorded per second and a 2D array where the first axis represents the recorded samples of amplitudes in the audio,the second axis represents the number of channels in the audio.
    # It is used to Load an audio file as a floating point time series
    # print(audio_data.shape)
    # print(sample_rate)
    # noise reduction
    noisy_part = audio_data[0:25000]  
    reduced_noise = nr.reduce_noise(audio_clip=audio_data, noise_clip=noisy_part, verbose=False)
    
    #trimming
    trimmed, index = librosa.effects.trim(reduced_noise, top_db=20, frame_length=512, hop_length=64)
    
    # extract features
    stft = np.abs(librosa.stft(trimmed, n_fft=512, hop_length=256, win_length=512)) 
    # print(stft)
    # save features
    # Features are saved as a numpy array(that's why it is saved as .npy file)
    np.save("STFT_features/stft_257_1/" + subject + "_" + name[:-4] + "_" + activity + ".npy", stft)

activities = ['Airplane', 'Breathing', 'BrushingTeeths', 'CanOpening', 'CarHorn', 'Cat', 'ChirpingBirds', 'ChurchBells', 'Clapping', 'ClockAlarm', 'ClockTick', 'Coughing', 'Cow', 'CracklingFire', 'Crow', 'CryingBaby', 'Dog', 'Door_or_WoodCreaks', 'DoorKnock', 'Drinking', 'Engine', 'Fireworks', 'FlyingInsects', 'Footsteps', 'Frog', 'GlassBreaking', 'HandSaw', 'Helicopter', 'Hen', 'KeyboardTyping', 'Laughing', 'MouseClick', 'Night', 'Pig', 'PouringWater', 'Rain', 'Rooster', 'SeaWaves', 'Sheep', 'Siren', 'Sneezing', 'Snoring', 'Thunderstorm', 'ToiletFlush', 'Train', 'TwoWheeler', 'VaccumCleaner', 'WashingMachine', 'WaterDrops', 'Wind']
    
subjects = ['s01', 's02', 's03', 's04', 's05']

for activity in activities:
    for subject in subjects:
        innerDir = subject + "/" + activity
        for file in os.listdir("Dataset_audio/"):
            if(file.endswith(".wav")):
                save_STFT("Dataset_audio/" + file, file, activity, subject)
                print(subject,activity,file)
