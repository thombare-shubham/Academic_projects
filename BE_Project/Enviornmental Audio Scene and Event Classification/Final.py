# Model Libraries
import os
import librosa
import datetime
import numpy as np
import pandas as pd
import noisereduce as nr
from keras import layers
from keras.utils import np_utils
from keras.models import Sequential,model_from_json
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense,Dropout,Activation
# Visualizatio Libraries
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import *
from tkinter import messagebox

from tensorflow.python.keras.backend import equal

main_ui = Tk()

activities = ['CanOpening', 'CarHorn', 'Cat', 'ChirpingBirds', 'Clapping', 'ClockAlarm', 'Cow', 'CracklingFire', 'Crow', 'CryingBaby', 'Dog', 'Door_or_WoodCreaks', 'Engine', 'Fireworks', 'GlassBreaking', 'HandSaw', 'Helicopter', 'Hen', 'Laughing', 'Night', 'Pig', 'Rain', 'Rooster', 'SeaWaves', 'Siren', 'Snoring', 'Thunderstorm', 'Train', 'TwoWheeler', 'VaccumCleaner', 'WaterDrops', 'Wind']

subjects = ['s01', 's02', 's03', 's04', 's05']

train_subjects = ['s01', 's02','s03']
validation_subjects = ['s04']
test_subjects = ['s05']

chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-" 
charsLen = len(chars)

# Replicate LableEncoder
# LabelEncoder is used for normalizing label values
lb = LabelEncoder()

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

# SAVE FEATURES
def Feature_Extraction():
    for activity in activities:
        for subject in subjects:
            innerDir = subject + "/" + activity
            for file in os.listdir("Dataset_audio/"+innerDir+"/"):
                if(file.endswith(".wav")):
                    save_STFT("Dataset_audio/"+innerDir+"/" + file, file, activity, subject)
                    print("Extracting feature from "+subject+"-"+file+"-"+activity)

    messagebox.showinfo("Features Extracted","Features of audio files are extracted!")

# CONVERT STRING TO NUMBER
def strToNumber(numStr):
  num = 0
  for i, c in enumerate(reversed(numStr)):
    num += chars.index(c) * (charsLen ** i)
  return(num)

# DIFFERENTIATE DATA IN TRAIN,TEST AND VALIDATION MODULE
def get_data(path):
        
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    X_validation = []
    Y_validation = []
        
    for file in os.listdir(path ):
        if int(strToNumber(file.split("_")[1].split("_")[0]))!=1:
          a = (np.load(path + file)).T
          label = file.split('_')[-1].split(".")[0]
          if(label in activities):
              #if(a.shape[0]>100 and a.shape[0]<500):
                if file.split("_")[0] in train_subjects:
                  X_train.append(np.mean(a,axis=0))
                  Y_train.append(label)
                elif file.split("_")[0] in validation_subjects:
                  X_validation.append(np.mean(a,axis=0))
                  Y_validation.append(label)
                else:
                  X_test.append(np.mean(a,axis=0))
                  Y_test.append(label)
                  
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_validation = np.array(X_validation)
    Y_validation = np.array(Y_validation)
    
    return X_train,Y_train,X_validation,Y_validation,X_test,Y_test

# BUILD AND SAVE MODEL
def Build_and_save_model():
    X_train,Y_train,X_validation,Y_validation,X_test,Y_test = get_data("Features/")

    n_samples = len(Y_train)
    messagebox.showinfo("Sample Count","No of samples to train: "+str(n_samples))
    order = np.array(range(n_samples))
    np.random.shuffle(order)
    X_train = X_train[order]
    Y_train = Y_train[order]

    lb = LabelEncoder()
    y_train = np_utils.to_categorical(lb.fit_transform(Y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(Y_test))
    y_validation = np_utils.to_categorical(lb.fit_transform(Y_validation))
    num_labels = y_train.shape[1]

    # BUILD MODEL
    model = Sequential()

    model.add(Dense(256, input_shape=(257,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))#Function used for regularization. regularization is the process which regularizes or shrinks the coefficients towards zero which reduces overfitting.

    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    model.fit(X_train, y_train, batch_size=8, epochs=200,validation_data=(X_validation,y_validation))

    prediction = model.predict(X_test)

    # Print Accuracy
    score = model.evaluate(X_train, y_train, verbose=0)
    print("Training Accuracy: {0:.2%}".format(score[1]))
    messagebox.showinfo("Accuracy","Model is giving accuracy of "+str(score)+"%")
    model_acc = model.evaluate(X_test, y_test, verbose=0)
    print("Testing Accuracy: {0:.2%}".format(model_acc[1]))

    # ## save model (optional)
    path = "Models/"+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    model_json = model.to_json()
    with open(path+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(path+".h5")
    messagebox.showinfo("Model Built","Model is ready")

# SPLIT AUDIO
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

def minMaxNormalize(arr):
    mn = np.min(arr)  # returns element wise minimum of array elements.
    mx = np.max(arr)  # returns element wise maximum of array elements.
    return (arr-mn)/(mx-mn)

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
    stfts = minMaxNormalize(stfts)
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
        noisy_part = raw_audio[0:50000]

        nr_audio = nr.reduce_noise(
            audio_clip=raw_audio, noise_clip=noisy_part, verbose=False)

        clip, index = librosa.effects.trim(nr_audio, top_db=20, frame_length=512, hop_length=64)
        result = predictSound(clip)
        res_prediction = str(result)
        if res_prediction in ("CanOpening" , "CarHorn" , "Clapping" , "ClockAlarm" , "Cow" , "Crow" , "CryingBaby" , "Dog" , "Engine" , "Fireworks" , "GlassBreaking" , "HandSaw" , "Helicopter" , "Laughing" , "Siren" , "Snoring" , "Thunderstorm" , "Train" , "TwoWheeler" , "VaccumCleaner"):
            messagebox.showwarning("Unhealthy Environment","It's sound of a "+result+" and it's noisy for children!")
        
        else:
            messagebox.showinfo("Healthy Environment","Children are in pretty much good environment!")
        
        del result

    else:
        messagebox.showinfo("Error","Wrong file selected/No file Selected")

# EXIT FUNCTION FOR SECOND WINDOW
def exit():
    main_ui.destroy()
    end_ui = Tk()
    end_ui.configure(bg='pink')
    end_ui.title("Thanking you!")
    end_ui.geometry('768x400')
    end_label = Label(end_ui,text='Presented By:\n\n1.Shubham Thombare\n\n2.Vishal Kajale\n\n3.Ankush Soni\n\n4.Kunal Sonar\n\n\nThank You!',bg='pink', fg='black', font=('opensans', 15, 'bold')).pack()
    MainButton2 = Button(end_ui, text="Exit", fg="pink", bg="purple", command=end_ui.destroy,border=0,width=15,
                         font=('arial', 12, 'italic')).place(x=330, y=350)

    end_ui.mainloop()

# PLOT AUDIO FILE
def plotaudio(output,label):
    fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    plt.title(label)
    plt.xlabel("No of Channels")
    plt.ylabel("Amplitude")
    plt.plot(output,color= "blue")
    ax.set_xlim((0,len(output)))
    ax.margins(2,-0.1)
    plt.show()

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

#FIRST WINDOW STARTS FROM HERE
main_ui = Tk()

main_ui.configure(bg='pink')

main_ui.title("Environmental Audio Scene and Sound Event Recogntion")
main_ui.geometry('850x550')

Wel_Label = Label(main_ui,text='Welcome Geek!\n', fg="black",bg="pink", font=('Arial', 20, 'bold')).pack()

MainLabel = Label(main_ui,text='Select the process\n\n1.Extract Features from Audio samples\n\n2.Build Model\n\n3.Check a sound file\n\n4.Visualize an audio file processing before feature extraction\n\n5.Exit\n', fg="black",bg="pink", font=('Arial', 12, 'bold')).pack()

feature_extr_button = Button(main_ui,text="Feature Extraction", fg="pink", bg="purple", command=Feature_Extraction, font=('opensans', 12, 'bold'),border=0,width=15).place(x=25, y=325)

model_build_button = Button(main_ui,text="Build Model", fg="pink", bg="purple", command=Build_and_save_model, font=('opensans', 12, 'bold'),border=0,width=15).place(x=350, y=325)

predict_button = Button(main_ui,text="Prediction", fg="pink", bg="purple", command=run_prediction, font=('opensans', 12, 'bold'),border=0,width=15).place(x=675, y=325)

visualizer_button = Button(main_ui,text="Visualize", fg="pink", bg="purple", command=noise_reduction, font=('opensans', 12, 'bold'),border=0,width=15).place(x=185, y=400)

exit_button = Button(main_ui,text='Exit', fg="pink", bg='purple',command=exit, font=('opensans', 12, 'bold'),border=0,width=15).place(x=515, y=400)

main_ui.mainloop()