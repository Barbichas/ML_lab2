#15.10.2024
#Machine Learning lab
#The goal is to identify craters in pictures of mars

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

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


####       Normalize       #################################
X_train_means = np.mean(X_train,axis = 0)    #Important for finale!
X_train_centered = X_train - X_train_means
X_train_centered_maxs = np.max(np.abs(X_train_centered), axis=0)  # Important for finale!
X_train_normalised = X_train_centered / X_train_centered_maxs
X_train_normalised_std_devs = np.std(X_train_centered, axis=0 ) #Important for finale!
X_train_normalised = X_train_normalised/ X_train_normalised_std_devs

y_train_mean = np.mean(y_train)          #Important for finale!
y_train_centered = y_train - y_train_mean
y_train_centered_max = np.max(y_train_centered)
y_train_normalised = y_train_centered / y_train_centered_max
y_train_normalised_std_dev = np.std(y_train_centered) #use standard deviation to normalise gaussian noise
y_train_normalised = y_train_normalised/ y_train_normalised_std_dev

def display_image(vector, L, H):
    # Convert the vector to a 2D array (image) with shape (H, L)
    image = np.array(vector).reshape(H, L)

    # Display the image using matplotlib
    plt.imshow(image, cmap='gray')  # Using 'gray' colormap for grayscale images
    plt.axis('off')  # Hide axes

###  look at some images  ###
if(0):
    for i in range(20):
        display_image(X_train[i],48,48)
        
        if(y_train[i]==1):
            plt.title("Image "+ str(i) + " has a cratter")
        else:
            plt.title("Image "+ str(i) + " is clean")
        plt.figure()

################################################################################
###############              Decision Tree
###############################################################################3
print()
print("----------------------------")
print("Random Decision Tree")
print()

random_state_max = 100

f1_train = []
f1_val = []
if (1):
    for rs in range(1,random_state_max):
        if(rs%10==0):
            print("Number of random states = " + str(rs) )
        clf = DecisionTreeClassifier(random_state=rs)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_train)
        TP = 0
        FN = 0
        FP = 0
        for i in y_pred:
            delta = y_train[i] - y_pred[i]
            if delta == 0:
                TP += 1
            elif delta == 1:
                FN += 1
            else:
                FP += 1
        F1 = 2*TP/(2*TP+FN+FP)
        f1_train.append(F1)
        #print("Number of errors is " + str(FN+FP) + " out of " + str(len(X_train)) + "images")
        #print("Training F1 is " + str(F1))

        y_pred = clf.predict(X_val)
        TP = 0
        FN = 0
        FP = 0
        for i in y_pred:
            delta = y_val[i] - y_pred[i]
            if delta == 0:
                TP += 1
            elif delta == 1:
                FN += 1
            else:
                FP += 1
        F1 = 2*TP/(2*TP+FN+FP)
        f1_val.append(F1)
        #print("Number of errors is " + str(FN+FP) + " out of " + str(len(X_val)) + "images")
        #print("Validation F1 is " + str(F1))

plt.plot(f1_train,color = "blue",label = "Train")
plt.plot(f1_val  ,color = "red" ,label = "Validation")
plt.legend()
plt.xlabel("Number of random states")
plt.ylabel("F1")
plt.title("F1 metric versus number of random states(Random Decision Tree)")
plt.show()

