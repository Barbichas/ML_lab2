#17.10.2024
#Machine Learning lab
#The goal is to identify craters in pictures of mars

#############################################

import json
import math
import numpy as np
from sklearn.metrics import f1_score

##################################################

import random
import matplotlib.pyplot as plt


X_train = np.load("Xtrain1.npy")
y_train = np.load("Ytrain1.npy")
X_test  = np.load("Xtest1.npy")

####################################################################
### define the validation and training sets  #######################
####################################################################
percent_val = 30
n_val = int(percent_val * len(X_train) / 100)

print("Use " + str(n_val) + " images for validation")
X_val = X_train[0:n_val]
y_val = y_train[0:n_val]

X_train = X_train[n_val:]
y_train = y_train[n_val:]


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
print("Number of validation images = " +str(X_val.shape[0]) + str(" ( ") + str(percent_val) + " %)")
print("Number of craters = " + str(n_craters))
print("Number of plain= " + str(n_plain))
print("Percentage of craters is " + str(100*n_craters/train_total) + " %")
print("Percentage of plain is " + str(100*n_plain/train_total) + " %")


####### Function to rotate image  ###########
def rotate_image_Sofia(image):
    for i in range(int(len(image)/2)):
        ii = len(image) -1 - i

        aux = image[i]
        image[i] = image[ii]
        image[ii] = aux
    return image

####### Function for brightness variations image  ###########
def bright_image(image):
    fator_brilho = 1.5
    image_bright = np.clip(image * fator_brilho, 0, 255)#.astype(np.uint8)

    #cv2.imshow('Imagem Original', image)
    #cv2.imshow('Imagem com Mais Brilho', image_bright)

    return image_bright

####### Function for transposing the image   ###########
def transpose_image(image):
    matrix = np.array(image).reshape(48, 48)
    transpose = np.transpose(matrix)
    return np.array(transpose).reshape(48**2)

#######  Function for negative           ############
def negative_image(image):
    negative = [255-pixel for pixel in image ]
    return np.array(negative)

########   Even number of craters and plains  ##############
def equalize_crat_and_plain(X, y):
    craters = list(X[y==1] )
    plains  = list(X[y==0] )

    while(1):
        
        if len(craters) == len(plains) :
            break
        #add random plain
        aux = plains[random.randint(0, len(plains)-1)]
        aux = rotate_image_Sofia(aux)
        plains.append(aux)

        if len(craters) == len(plains) :
            break
        #add random plain 
        aux = plains[random.randint(0, len(plains)-1)]
        aux = bright_image(aux)
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

def add_transpose_images(X,y):
    more_X = []
    for image in X:
        more_X.append(transpose_image(image))
    more_X = np.array(more_X)
    X_final = np.concatenate( (X,more_X) )
    y_final = np.concatenate( (y,y) )

    return X_final , y_final

def add_bright_images(X,y):
    more_X = []
    for image in X:
        more_X.append(bright_image(image))
    more_X = np.array(more_X)
    X_final = np.concatenate( (X,more_X) )
    y_final = np.concatenate( (y,y) )

    return X_final , y_final

def add_negative_images(X,y):
    more_X = []
    for image in X:
        more_X.append(bright_image(image))
    more_X = np.array(more_X)
    X_final = np.concatenate( (X,more_X) )
    y_final = np.concatenate( (y,y) )

    return X_final , y_final


print("Equalize number of images")
X_train, y_train = equalize_crat_and_plain(X_train,y_train)
#X_train, y_train = add_transpose_images(X_train,y_train)
#X_train, y_train = add_bright_images(X_train, y_train)
#X_train , y_train = add_negative_images(X_train, y_train)

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


##########  Prepare shapes for CNN  ##########
X_train = np.array(X_train).reshape(X_train.shape[0],48 , 48)
X_val   = np.array(X_val ).reshape(X_val.shape[0]  ,48 , 48)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], X_val.shape[2], 1))

################################################################################
###############              CNN
################################################################################

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam

# Initialize the CNN model
model = Sequential()
train_and_save = 1      ### se quiseres treinar

if (train_and_save):
    # 1st Convolutional Layer + Pooling
    model.add(Conv2D(32, (3, 3), input_shape=(48, 48, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 2nd Convolutional Layer + Pooling
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 3rd Convolutional Layer + Pooling
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the layers
    model.add(Flatten())

    # Fully connected layer (Dense layer)
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout for regularization

    # Output layer (Sigmoid for binary classification)
    model.add(Dense(units=1, activation='sigmoid'))

    # Compile the model
    learning_rate = 0.0005  # Set your desired learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        steps_per_epoch=len(X_train),
        epochs=10,  # Number of epochs (adjust as needed)
        validation_data=(X_val , y_val),
        validation_steps=len(X_val)
    )
    # Save the model
    model.save('cnn_binary_classifier.h5')

else:
    model = load_model('cnn_binary_classifier.h5')

# Evaluate the model on the validation/test set
test_loss, test_acc = model.evaluate(X_val , y_val)
print(f"Test Accuracy: {test_acc}")

# Make predictions
y_val_pred = model.predict(X_val)
# Convert predictions to binary values (0 or 1)
y_val_pred = (y_val_pred > 0.5).astype(int)

# Calculate F1 score
f1 = f1_score(y_val, y_val_pred)
print(f"Learning rate: {learning_rate} Validation F1 Score: {f1}")








