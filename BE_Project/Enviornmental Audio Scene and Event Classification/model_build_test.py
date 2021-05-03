# Code to build model
from sklearn.preprocessing import LabelEncoder
# LabelEncoder can be used to normalize labels.It can also be used to transform non-numerical labels (as long as they are hashable and comparable) to numerical labels.
import numpy as np ##Numpy is a python library that provides a simple yet powerful data structure:the n-dimensional array. This is the foundation on which almost all power of python's data science toolkit is built and learning NumPy is the first step on any python data scientist journey.
import os ##Provides functions for interacting with operating system
from keras.models import Sequential #A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
from keras.layers import Dense,Dropout,Activation #Layers to be used in training model
from keras.utils import np_utils #
import datetime #Module to work with dates
from sklearn import metrics
from tensorflow.python.keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint 


# CONVERT STRING TO INTEGER
chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
charslen = len(chars)

def strToNumber(numStr):
    num = 0
    for i,c in enumerate(reversed(numStr)):
        num += chars.index(c) * (charslen ** i)
    return(num)

# #RESAMPLE DATA
# def reSample(data, samples):
#     r = len(data)/samples #resampling ratio
#     newdata = []
#     for i in range(0,samples):
#         newdata.append(data[int(i*r)])
#     return np.array(newdata)

train_subjects = ['s01','s02','s03']
validation_subjects = ['s04']
test_subjects = ['s05']

def get_data(path,samplesize):
    Activities = ['Airplane','CanOpening','CarHorn','Clapping','ClockAlarm','Crow','CryingBaby','Dog','Engine','Fireworks','GlassBreaking','HandSaw','Helicopter','Laughing','Siren','Thunderstorm','Train','VaccumCleaner','Other']

    X_train = []
    Y_train = []
    X_validation = []
    Y_validation = []
    X_test = []
    Y_test = []

    for file in os.listdir(path ):
        if int(strToNumber(file.split("_")[1].split("_")[0]))!=1:
            a = (np.load(path + file)).T #.T is used for making transpose of matrix
            label = file.split('_')[-1].split(".")[0]
            if(label in Activities):
                    if file.split("_")[0] in train_subjects:
                        X_train.append(np.mean(a,axis=0))
                        # a is Array containing numbers whose mean is desired. If a is not an array, a conversion is attempted.
                        # Compute the arithmetic mean along the specified axis here it's axis=0 i.e compute alongside rows.
                        Y_train.append(label)
                        # X contains mean value while y contains label

                    elif file.split("_")[0] in validation_subjects:
                        X_validation.append(np.mean(a,axis=0))
                        Y_validation.append(label)
                    else:
                        X_test.append(np.mean(a,axis=0))
                        Y_test.append(label)

    #Convert data stored in array to numpy array
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_validation = np.array(X_validation)
    Y_validation = np.array(Y_validation)

    return X_train,Y_train,X_validation,Y_validation,X_test,Y_test

X_train,Y_train,X_validation,Y_validation,X_test,Y_test = get_data("STFT_features/",250)

n_samples = len(Y_train)
print("No of training samples: "+ str(n_samples))
# order = np.array(range(n_samples))
# np.random.shuffle(order)
# X_train = X_train[order]
# Y_train = Y_train[order]

lb = LabelEncoder()#LabelEncoder encode labels with a value between 0 and n_classes-1 where n is the number of distinct labels. If a label repeats it assigns the same value to as assigned earlier.
y_train = np_utils.to_categorical(lb.fit_transform(Y_train)) 
# fit_transform - Fit label encoder and return encoded labels.
# np_utils.to_categorical - Converts a class vector (integers) to binary class matrix.
y_test = np_utils.to_categorical(lb.fit_transform(Y_test))
y_validation = np_utils.to_categorical(lb.fit_transform(Y_validation))
num_labels = y_train.shape[1] #shape[1] to calculate no of columns
filter_size = 2

# BUILD MODEL
model = Sequential()
model.add(Dense(256,input_shape=(X_train.shape[1],)))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
# Dense depicts Just your regular densely-connected NN layer.
# Relu is activation function - The rectified linear activation function or ReLU for short is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero.With default values, this returns the standard ReLU activation: max(x, 0), the element-wise maximum of 0 and the input tensor.

model.add(Dense(256))
model.add(Activation('relu'))
# model.add(Dropout(0.5))

model.add(Dense(128))
model.add(Activation('relu'))
# model.add(Dropout(0.5))

# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))#Function used for regularization. Regularization is a process which regularizes or shrinks the coefficients towards zero which reduces overfitting.

model.add(Dense(num_labels))
model.add(Activation('softmax'))
# Softmax converts a real vector to a vector of categorical probabilities.Softmax is often used as the activation for the last layer of a classification network because the result could be interpreted as a probability distribution.The softmax of each vector x is computed as exp(x) / tf.reduce_sum(exp(x)).
model.compile(loss = 'categorical_crossentropy',metrics=['accuracy'],optimizer='adam')#Configures the model for training.
model.summary()

model.fit(X_train,y_train,batch_size = 10, epochs=60,validation_data=(X_validation,y_validation))
# Trains the model for a fixed number of epochs (iterations on a dataset).
result = model.predict(X_test)
# This is called a probability prediction where, given a new instance, the model returns the probability for each outcome class as a value between 0 and 1.

# Evaluating the model on the training and testing set
score = model.evaluate(X_train, y_train, verbose=0)
print("Training Accuracy: {0:.2%}".format(score[1]))
model_acc = model.evaluate(X_test, y_test, verbose=0)
print("Testing Accuracy: {0:.2%}".format(model_acc[1]))

# SAVE MODEL
# path = "Models/"+ datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
# model_json = model.to_json()
# with open(path+str(model_acc)+".json","w") as json_file:
#     json_file.write(model_json)

# Serialize weights to HDF5
# model.save_weights(path+str(model_acc)+".h5")
# HDF5 = The Hierarchical Data Format version 5 (HDF5), is an open source file format that supports large, complex, heterogeneous data. HDF5 uses a "file directory" like structure that allows you to organize data within the file in many different structured ways, as you might do with files on your computer