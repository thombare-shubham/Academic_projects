from sklearn.preprocessing import LabelEncoder
# import pandas as pd
import numpy as np
import os
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Dropout, Activation, Flatten,LSTM,TimeDistributed
# from keras.layers import Convolution2D, MaxPooling2D,MaxPooling1D,Conv1D
# from keras.optimizers import Adam,SGD
from keras.utils import np_utils
# from sklearn import metrics
import random
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras import optimizers
import datetime

# Convert String to Integer
chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-" 
charsLen = len(chars)
def strToNumber(numStr):
  num = 0
  for i, c in enumerate(reversed(numStr)):
    num += chars.index(c) * (charsLen ** i)
  return(num)

train_subjects = ['s01', 's02','s03']
validation_subjects = ['s04']
test_subjects = ['s05']

def get_data(path):
    
    activities = ['CanOpening', 'CarHorn', 'Cat', 'ChirpingBirds', 'Clapping', 'ClockAlarm', 'Cow', 'CracklingFire', 'Crow', 'CryingBaby', 'Dog', 'Door_or_WoodCreaks', 'Engine', 'Fireworks', 'GlassBreaking', 'HandSaw', 'Helicopter', 'Hen', 'Laughing', 'Night', 'Pig', 'Rain', 'Rooster', 'SeaWaves', 'Sheep', 'Siren', 'Snoring', 'Thunderstorm', 'Train', 'TwoWheeler', 'VaccumCleaner', 'WaterDrops', 'Wind']
    
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

X_train,Y_train,X_validation,Y_validation,X_test,Y_test = get_data("Features/")

n_samples = len(Y_train)
print("No of training samples: " + str(n_samples))
order = np.array(range(n_samples))
np.random.shuffle(order)
X_train = X_train[order]
Y_train = Y_train[order]

lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(Y_train))
y_test = np_utils.to_categorical(lb.fit_transform(Y_test))
y_validation = np_utils.to_categorical(lb.fit_transform(Y_validation))
num_labels = y_train.shape[1]

num_labels = y_train.shape[1]
filter_size = 2

# build model
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

model.fit(X_train, y_train, batch_size=8, epochs=100,validation_data=(X_validation,y_validation))

prediction = model.predict(X_test)

# Print Accuracy
score = model.evaluate(X_train, y_train, verbose=0)
print("Training Accuracy: {0:.2%}".format(score[1]))
model_acc = model.evaluate(X_test, y_test, verbose=0)
print("Testing Accuracy: {0:.2%}".format(model_acc[1]))

# ## save model (optional)
path = "Models/"+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
model_json = model.to_json()
with open(path+".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(path+".h5")