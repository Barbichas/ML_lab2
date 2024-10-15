#13.10.2024
#Machine Learning lab
#The goal is to identify craters in pictures of mars

#############################################
#import os

#os.environ["KERAS_BACKEND"] = "tensorflow"  # @param ["tensorflow", "jax", "torch"]

import json
import math
import numpy as np

import keras
from keras import losses
from keras import ops
from keras import optimizers
from keras.optimizers import schedules
from keras import metrics

import keras_cv

# Import tensorflow for [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) and its preprocessing functions
import tensorflow as tf
import tensorflow_datasets as tfds
##################################################

import random
import matplotlib.pyplot as plt


X_train = np.load("Xtrain1.npy")
y_train = np.load("Ytrain1.npy")
X_test  = np.load("Xtest1.npy")

########    Counting labels    ##########
train_total = y_train.shape[0]
n_craters = 0
n_plain   = 0
for element in y_train:
    if element== 0:
        n_plain += 1
    else: n_craters += 1

print()
print("DESCRIPTION OF DATASET")
print()
print("Number of training images = " + str(train_total))
print("Number of craters = " + str(n_craters))
print("Number of plain= " + str(n_plain))
print("Percentage of craters is " + str(100*n_craters/train_total) + " %")
print("Percentage of plain is " + str(100*n_plain/train_total) + " %")

####### Function to rotate image  ###########
def flip_image_Sofia(image):
    for i in range(int(len(image)/2)):
        ii = len(image) -1 - i

        aux = image[i]
        image[i] = image[ii]
        image[ii] = aux
    return image

########   Even number of craters and plains  ##############
def equalize_crat_and_plain(X, y):
    craters = list(X[y==1] )
    plains  = list(X[y==0] )

    while(1):

        if len(craters) == len(plains) :
            break
        #remove random crater
        craters.pop(random.randint(0,len(craters)-1))
        
        if len(craters) == len(plains) :
            break
        #add random plain
        aux = plains[random.randint(0, len(plains)-1)]
        aux = flip_image_Sofia(aux)
        plains.append(aux)

    #add labels to the data
    for i in range(len(craters)):
        craters[i] = np.concatenate( ([1] , craters[i]) )
        plains[i]  = np.concatenate( ([0] , plains[i] ) )
    X_final = craters + plains 
    random.shuffle(X_final)

    y_final = []
    for i in range( len(X_final) ):
        y_final.append( X_final[i][0] ) #put the label in y
        X_final[i]=X_final[i][1:]       #remove the label from X
    
    X_final = np.array(X_final)
    y_final = np.array(y_final)

    return X_final, y_final

print("Equalize number of images")
X_train, y_train = equalize_crat_and_plain(X_train,y_train)

### define the validation and training sets  ########
percent_val = 5
n_val = int(percent_val * len(X_train) / 100)

print("Use " + str(n_val) + " images for validation")
X_val = X_train[0:n_val]
y_val = y_train[0:n_val]

X_train = X_train[n_val:]
y_train = y_train[n_val:]

################   Recount      ##################
train_total = y_train.shape[0]
n_craters = 0
n_plain   = 0
for element in y_train:
    if element== 0:
        n_plain += 1
    else: n_craters += 1

print("-------------------------------------------------------------")
print("DESCRIPTION OF BALANCED DATASET")
print()
print("Number of test images       = " +str(X_test.shape[0]))
print("Number of validation images = " +str(X_val.shape[0]))
print("Number of training images = " + str(train_total))
print("Number of craters = " + str(n_craters))
print("Number of plain= " + str(n_plain))
print("Percentage of craters is " + str(100*n_craters/train_total) + " %")
print("Percentage of plain is " + str(100*n_plain/train_total) + " %")


################################################################################
###############              CNN
###############################################################################3




