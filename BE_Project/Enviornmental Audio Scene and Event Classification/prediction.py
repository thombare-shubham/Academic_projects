from tkinter import filedialog,messagebox
from keras.models import model_from_json
import numpy as np
import librosa
import noisereduce as nr

# Import files
from definations import *

# PREDICT SOUND
def predictSound(X):
    model_path = r"Models/"
    model_name = "2021_03_05_16_36_55"

    # Model reconstruction from a json file
    with open(model_path + model_name + '.json', 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights(model_path + model_name + '.h5')

    # using fit_transform
    lb.fit_transform((activities))
    # Returns magnitude of frequency bin f at frame t
    stfts = np.abs(librosa.stft(X, n_fft=512, hop_length=256, win_length=512))
    stfts = np.mean(stfts, axis=1)
    result = model.predict(np.array([stfts]))  # Predict Output of model
    predictions = [np.argmax(y) for y in result]
    return lb.inverse_transform([predictions[0]])[0]

# FUNCTION TO RUN PREDICTION
def run_prediction():
# Import file using tkinter
    main_ui.filename = filedialog.askopenfilename(
       initialdir="/", filetypes=(("Audio files", "*.wav"), ("all files", "*.*")))

    if main_ui.filename.endswith('.wav'):

        raw_audio, sr = librosa.load(main_ui.filename)
        noisy_part = raw_audio[0:25000]

        nr_audio = nr.reduce_noise(
            audio_clip=raw_audio, noise_clip=noisy_part, verbose=False)

        clip, index = librosa.effects.trim(nr_audio, top_db=20, frame_length=512, hop_length=64)
        result = predictSound(clip)
        res_prediction = str(result)
        if res_prediction in ("CanOpening" , "CarHorn" , "Clapping" , "ClockAlarm" , "Cow" , "Crow" , "CryingBaby" , "Dog" , "Engine" , "Fireworks" , "GlassBreaking" , "HandSaw" , "Helicopter" , "Laughing" , "Siren" , "Snoring" , "Thunderstorm" , "Train" , "TwoWheeler" , "VaccumCleaner"):
            messagebox.showwarning("Unhealthy Environment","It's sound of a "+result+" and it's noisy for children!")
        
        else:
            messagebox.showinfo("Healthy Environment","It's sound of a"+result+" & Its not noisy for children!")
        
        del result

    else:
        messagebox.showinfo("Error","Wrong file selected/No file Selected")
