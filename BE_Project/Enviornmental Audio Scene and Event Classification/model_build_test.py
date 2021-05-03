from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed
from keras.layers import Convolution2D, MaxPooling2D, MaxPooling1D, Conv1D
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from sklearn import metrics
import random
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers, layers
import datetime

# Convert String to Integer
chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
charsLen = len(chars)


def strToNumber(numStr):
    num = 0
    for i, c in enumerate(reversed(numStr)):
        num += chars.index(c) * (charsLen ** i)

    return(num)


train_subjects = ['s01', 's02', 's03']
test_subjects = ['s04', 's05']


def get_data(path):
    Activities = ['Airplane', 'CanOpening', 'CarHorn', 'Cat', 'ChirpingBirds', 'ChurchBells', 'Clapping', 'ClockAlarm', 'Coughing', 'Cow', 'CracklingFire', 'Crow', 'CryingBaby', 'Dog', 'Door_or_WoodCreaks', 'DoorKnock', 'Engine', 'Fireworks',
        'Frog', 'GlassBreaking', 'HandSaw', 'Helicopter', 'Hen', 'Laughing', 'Night', 'Pig', 'PouringWater', 'Rain', 'Rooster', 'SeaWaves', 'Sheep', 'Siren', 'Sneezing', 'Thunderstorm', 'Train', 'VaccumCleaner', 'WaterDrops', 'Wind']

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    for file in os.listdir(path+"stft_257_1/"):
        if int(strToNumber(file.split("_")[1].split("_")[0])) != 1:
            a = (np.load(path + "stft_257_1/" + file)).T
            label = file.split('_')[-1].split(".")[0]
            if(label in Activities):
                if file.split("_")[0] in train_subjects:
                    X_train.append(np.mean(a, axis=0))
                    Y_train.append(label)
                else:
                    X_test.append(np.mean(a, axis=0))
                    Y_test.append(label)

    # Convert data stored in array to numpy array
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    return X_train, Y_train, X_test,Y_test

X_train, Y_train, X_test,Y_test = get_data("STFT_features/")

n_samples = len(Y_train)
print("No of training samples: " + str(n_samples))

lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(Y_train))
y_test = np_utils.to_categorical(lb.fit_transform(Y_test))
num_labels = y_train.shape[1]  # shape[1] to calculate no of columns
filter_size = 2

# BUILD MODEL
model = Sequential()
model.add(layers.Dense(256, activation='relu',
                       input_shape=(X_train.shape[1],)))
# model.add(Dropout(0.5))

model.add(layers.Dense(256, activation='relu'))
# model.add(Dropout(0.5))

model.add(layers.Dense(128, activation='relu'))
# model.add(Dropout(0.5))

model.add(layers.Dense(128, activation='relu'))
# model.add(Dropout(0.5))

model.add(layers.Dense(64, activation='relu'))
# model.add(Dropout(0.5))

model.add(layers.Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(layers.Dense(num_labels, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# model.summary()

model.fit(X_train, y_train, batch_size = 8, epochs=10000)
result = model.predict(X_test)

# Evaluating the model on the training and testing set
score = model.evaluate(X_train, y_train, verbose=0)
print("Training Accuracy: {0:.2%}".format(score[1]))
model_acc = model.evaluate(X_test, y_test, verbose=0)
print("Testing Accuracy: {0:.2%}".format(model_acc[1]))
