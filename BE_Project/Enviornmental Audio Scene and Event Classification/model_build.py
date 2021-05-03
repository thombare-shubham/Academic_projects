from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,LSTM,TimeDistributed
from keras.layers import Convolution2D, MaxPooling2D,MaxPooling1D,Conv1D
from keras.optimizers import Adam,SGD
from keras.utils import np_utils
from sklearn import metrics
import random
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
import datetime

# Convert String to Integer
chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-" 
charsLen = len(chars)

def strToNumber(numStr):
  num = 0
  for i, c in enumerate(reversed(numStr)):
    num += chars.index(c) * (charsLen ** i)

  return(num)

#Resample Data 
def reSample(data, samples):
    r = len(data)/samples #re-sampling ratio
    newdata = []
    for i in range(0,samples):
        newdata.append(data[int(i*r)])
    return np.array(newdata)
  
  
train_subjects = ['s01', 's02', 's03']
validation_subjects = ['s04']
test_subjects = ['s05']

def get_data(path,sampleSize):
    
    Negative_Activities = ['Clapping', 'ClockAlarm', 'ClockTick', 'CarHorn', 'CracklingFire', 'Crow', 'CryingBaby', 'Door_or_WoodCreaks', 'DoorKnock', 'Drinking', 'Engine','Fireworks', 'Footsteps', 'GlassBreaking', 'HandSaw', 'Helicopter', 'KeyboardTyping', 'Laughing', 'PouringWater', 'Siren', 'Snoring', 'ToiletFlush', 'Train', 'TwoWheeler', 'VaccumCleaner', 'WashingMachine','WaterDrops']
    
    # UnhealthyActivities = ['Coughing','Breathing', 'Sneezing']
    
    Positive_Activities = ['Airplane','Cat','ChirpingBirds','Cow', 'Dog', 'FlyingInsects', 'Frog', 'Hen', 'Night', 'Pig', 'Rain', 'Rooster', 'SeaWaves', 'Sheep', 'Thunderstorm', 'Wind']
    
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    X_validation = []
    Y_validation = []
        
    for file in os.listdir(path + 'stft_257_1/'):
        if int(strToNumber(file.split("_")[1].split("_")[0]))!=1:
          a = (np.load(path + "stft_257_1/" + file)).T
          label = file.split('_')[-1].split(".")[0]
          if(label in Positive_Activities):
                label = "Positive_Activities"
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

          elif(label in Negative_Activities):
                label = "Negative_Activities"
              
                if file.split("_")[0] in train_subjects:
                  X_train.append(np.mean(a,axis=0))
                  Y_train.append(label)
                elif file.split("_")[0] in validation_subjects:
                  X_validation.append(np.mean(a,axis=0))
                  Y_validation.append(label)
                else:
                  X_test.append(np.mean(a,axis=0))
                  Y_test.append(label)
          # elif(label in UnhealthyActivities):
          #       label = "UnhealthyActivities"
              
          #       if file.split("_")[0] in train_subjects:
          #         X_train.append(np.mean(a,axis=0))
          #         Y_train.append(label)
          #       elif file.split("_")[0] in validation_subjects:
          #         X_validation.append(np.mean(a,axis=0))
          #         Y_validation.append(label)
          #       else:
          #         X_test.append(np.mean(a,axis=0))
          #         Y_test.append(label)
                  
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_validation = np.array(X_validation)
    Y_validation = np.array(Y_validation)
    
    return X_train,Y_train,X_validation,Y_validation,X_test,Y_test

featuresPath = "STFT_features/"
a,b,c,d,e,f = get_data(featuresPath,250)

X_train,Y_train,X_validation,Y_validation,X_test,Y_test = a,b,c,d,e,f

n_samples = len(Y_train)
print("No of training samples: " + str(n_samples))
# order = np.array(range(n_samples))
# np.random.shuffle(order)
# X_train = X_train[order]
# Y_train = Y_train[order]

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
# model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
# model.add(Dropout(0.5))

model.add(Dense(128))
model.add(Activation('relu'))
# model.add(Dropout(0.5))

# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))#Function used for regularization. regularization is the process which regularizes or shrinks the coefficients towards zero which reduces overfitting.

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# model.summary()

model.fit(X_train, y_train, batch_size=10, epochs=60,validation_data=(X_validation,y_validation))

result = model.predict(X_test)

# Print Accuracy
score = model.evaluate(X_train, y_train, verbose=0)
print("Training Accuracy: {0:.2%}".format(score[1]))
model_acc = model.evaluate(X_test, y_test, verbose=0)
print("Testing Accuracy: {0:.2%}".format(model_acc[1]))

# showResult()

# ## save model (optional)
path = "Models/audio_NN_New"+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
model_json = model.to_json()
with open(path+"_acc_"+".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(path+"_acc_"+".h5")