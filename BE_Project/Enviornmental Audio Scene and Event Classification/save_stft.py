import librosa 
import noisereduce as nr
import numpy as np

# EXTRACT AND SAVE STFT FEATURES
def save_STFT(file, name, activity, subject):
    # read audio data
    audio_data, sample_rate = librosa.load(file)
    noisy_part = audio_data[0:25000]  
    reduced_noise = nr.reduce_noise(audio_clip=audio_data, noise_clip=noisy_part, verbose=False)
    
    #trimming
    trimmed, index = librosa.effects.trim(reduced_noise, top_db=20, frame_length=512, hop_length=64)
    
    # extract features
    stft = np.abs(librosa.stft(trimmed, n_fft=512, hop_length=256, win_length=512)) 
    # save features
    np.save("Features/" + subject + "_" + name[:-4] + "_" + activity + ".npy", stft)