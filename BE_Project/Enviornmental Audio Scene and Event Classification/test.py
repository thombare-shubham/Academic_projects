import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import os
import csv
# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
#Keras
import keras
from keras import models
from keras import layers
from keras.models import Sequential
import datetime
import noisereduce as nr

count = 0
activity_count = 0
print("Extrating Features...\n")
# generating a dataset
header = 'filename chroma_stft  spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

file = open('data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
    
Activities = 'Cat ChirpingBirds ChurchBells Cow CracklingFire Crow Frog Hen Night Rain Rooster SeaWaves Sheep Thunderstorm WaterDrops Wind'.split()
for Activity in Activities:
    activity_count+=1
    for filename in os.listdir(f'./Data/{Activity}'):
        Activity_name = f'./Data/{Activity}/{filename}'
        print("Extracting features from "+Activity_name)
        if Activity_name.endswith(".wav"):
            y, sr = librosa.load(Activity_name)
            noisy_part = y[0:25000] #Section of audio file i.e noise
            reduced_noise = nr.reduce_noise(audio_clip=y,noise_clip = noisy_part, verbose=False)

            # Trimming - Trim leading and trailing silence from an audio signal.
            trimmed, index = librosa.effects.trim(reduced_noise, top_db = 20, frame_length = 512, hop_length = 64)
            y = trimmed

        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        # rmse = librosa.feature.rmse(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {Activity}'
        file = open('data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
        count += 1

print("Feature extraction finished!\n")
print("Total "+str(count)+" files readed in "+ str(activity_count)+" activities!\n")
# reading dataset from csv

data = pd.read_csv('data.csv')
data.head()

# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)
data.head()

# Encode Activities into integers
activities_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(activities_list)
print(y)

# normalizing
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

# spliting of dataset into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)


# creating a model
model = Sequential()

model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(64,activation = 'relu'))

model.add(layers.Dense(activity_count+1, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
              
history = model.fit(X_train,
                    y_train,
                    epochs=100,
                    batch_size=8)
                    
# Evaluating the model on the training and testing set
score = model.evaluate(X_train, y_train, verbose=0)
print("Training Accuracy: {0:.2%}".format(score[1]))
model_acc = model.evaluate(X_test, y_test, verbose=0)
print("Testing Accuracy: {0:.2%}".format(model_acc[1]))

# predictions
result = model.predict(X_test)
# np.argmax(predictions[0])
# SAVE MODEL
path = "Models/"+ datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
model_json = model.to_json()
with open(path+".json","w") as json_file:
    json_file.write(model_json)

# Serialize weights to HDF5
model.save_weights(path+".h5")

print("Model Saved!")