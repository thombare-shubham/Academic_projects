# Code to build model
from sklearn.preprocessing import LabelEncoder
# LabelEncoder can be used to normalize labels.It can also be used to transform non-numerical labels (as long as they are hashable and comparable) to numerical labels.
import numpy as np ##Numpy is a python library that provides a simple yet powerful data structure:the n-dimensional array. This is the foundation on which almost all power of python's data science toolkit is built and learning NumPy is the first step on any python data scientist journey.
import os ##Provides functions for interacting with operating system
from keras.models import Sequential #A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor. #Layers to be used in training model
from keras.utils import np_utils #
import datetime #Module to work with dates
from sklearn import metrics
from tensorflow.python.keras.callbacks import Callback
from keras import layers 


# CONVERT STRING TO INTEGER
chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
charslen = len(chars)

def strToNumber(numStr):
    num = 0
    for i,c in enumerate(reversed(numStr)):
        num += chars.index(c) * (charslen ** i)
    return(num)

train_subjects = ['s01','s02','s03']
test_subjects = ['s04','s05']

def get_data(path):
    Activities = ['Airplane','CanOpening','CarHorn','Cat','ChirpingBirds','ChurchBells','Clapping','ClockAlarm','Coughing','Cow','CracklingFire','Crow','CryingBaby','Dog','Door_or_WoodCreaks','DoorKnock','Engine','Fireworks','Frog','GlassBreaking','HandSaw','Helicopter','Hen','Laughing','Night','Pig','PouringWater','Rain','Rooster','SeaWaves','Sheep','Siren','Sneezing','Thunderstorm','Train','VaccumCleaner','WaterDrops','Wind']

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    for file in os.listdir(path):
        if int(strToNumber(file.split("_")[1]))!=1:
            a = (np.load(path + file)).T #.T is used for making transpose of matrix
            label = file.split('_')[-1].split(".")[0]
            if(label in Activities):
                    if file.split("_")[0] in train_subjects:
                        X_train.append(np.mean(a,axis=0))
                        Y_train.append(label)
                    else:
                        X_test.append(np.mean(a,axis=0))
                        Y_test.append(label)

    #Convert data stored in array to numpy array
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    return X_train,Y_train,X_test,Y_test

X_train,Y_train,X_test,Y_test = get_data("Features/")

n_samples = len(Y_train)
print("No of training samples: "+ str(n_samples))

lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(Y_train)) 
y_test = np_utils.to_categorical(lb.fit_transform(Y_test))
num_labels = y_train.shape[1] #shape[1] to calculate no of columns
filter_size = 2

# BUILD MODEL
model = Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
# model.add(Dropout(0.5))

model.add(layers.Dense(256, activation='relu'))
# model.add(Dropout(0.5))

model.add(layers.Dense(128, activation='relu'))
# model.add(Dropout(0.5))

model.add(layers.Dense(num_labels, activation='softmax'))
model.compile(loss = 'categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
# model.summary()

model.fit(X_train,y_train,batch_size = 10, epochs=60)
result = model.predict(X_test)

# Evaluating the model on the training and testing set
score = model.evaluate(X_train, y_train, verbose=0)
print("Training Accuracy: {0:.2%}".format(score[1]))
model_acc = model.evaluate(X_test, y_test, verbose=0)
print("Testing Accuracy: {0:.2%}".format(model_acc[1]))

# SAVE MODEL
path = "Models/"+ datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
model_json = model.to_json()
with open(path+str(model_acc)+".json","w") as json_file:
    json_file.write(model_json)

# Serialize weights to HDF5
model.save_weights(path+str(model_acc)+".h5")
