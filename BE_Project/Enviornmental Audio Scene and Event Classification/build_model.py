from keras.layers import Dense,Dropout,Activation,Flatten
from keras.utils import np_utils
from keras.models import Sequential
from tkinter import messagebox
import numpy as np
import datetime

# Import files
from definations import *
from get_data import *

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

    model.add(Dense(256, input_shape=(257,), activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(64,activation='relu'))
    model.add(Flatten())#Function used for regularization. regularization is the process which regularizes or shrinks the coefficients towards zero which reduces overfitting.

    model.add(Dense(num_labels,activation='softmax'))

    model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer='adam')

    model.fit(X_train, y_train, batch_size=8, epochs=200,validation_data=(X_validation,y_validation))

    prediction = model.predict(X_test)
    # print(prediction)

    # Print Accuracy
    score = model.evaluate(X_train, y_train, verbose=0)
    print("Accuracy: {0:.2%}".format(score[1]))

    # # ## save model (optional)
    path = "Models/"+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    model_json = model.to_json()
    with open(path+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(path+".h5")

    messagebox.showinfo("Model Ready","Model is ready! Accuracy of model is "+str(score[1])+"%")
