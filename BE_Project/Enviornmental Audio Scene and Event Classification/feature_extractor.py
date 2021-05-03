import librosa
import os
import numpy as np
import noisereduce as nr


def save_STFT(file, name, activity, subject):
    # read audio data
    audio_data, sample_rate = librosa.load(file)
    
    # noise reduction
    noisy_part = audio_data[0:25000]  
    reduced_noise = nr.reduce_noise(audio_clip=audio_data, noise_clip=noisy_part, verbose=False)
    
    #trimming
    trimmed, index = librosa.effects.trim(reduced_noise, top_db=20, frame_length=512, hop_length=64)
    
    # extract features
    stft = np.abs(librosa.stft(trimmed, n_fft=512, hop_length=256, win_length=512))
    # save features
    np.save("STFT_features/stft_257_1/" + subject + "_" + name[:-4] + "_" + activity + ".npy", stft)

activities = ['Airplane', 'Breathing', 'BrushingTeeths', 'CanOpening', 'CarHorn', 'Cat', 'ChirpingBirds', 'ChurchBells', 'Clapping', 'ClockAlarm', 'ClockTick', 'Coughing', 'Cow', 'CracklingFire', 'Crow', 'CryingBaby', 'Dog', 'Door_or_WoodCreaks', 'DoorKnock', 'Drinking', 'Engine', 'Fireworks', 'FlyingInsects', 'Footsteps', 'Frog', 'GlassBreaking', 'HandSaw', 'Helicopter', 'Hen', 'KeyboardTyping', 'Laughing', 'MouseClick', 'Night', 'Pig', 'PouringWater', 'Rain', 'Rooster', 'SeaWaves', 'Sheep', 'Siren', 'Sneezing', 'Snoring', 'Thunderstorm', 'ToiletFlush', 'Train', 'TwoWheeler', 'VaccumCleaner', 'WashingMachine', 'WaterDrops', 'Wind']
    
subjects = ['s01', 's02', 's03', 's04', 's05']

for activity in activities:
    for subject in subjects:
        innerDir = subject + "/" + activity
        for file in os.listdir("Dataset_audio/" + innerDir):
            if(file.endswith(".wav")):
                save_STFT("Dataset_audio/" + innerDir + "/" + file, file, activity, subject)
                print(subject,activity,file)
