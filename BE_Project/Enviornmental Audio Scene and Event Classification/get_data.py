import numpy as np
import os

# import files
from definations import *

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