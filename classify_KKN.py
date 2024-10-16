#13.10.2024
#Machine Learning lab
#The goal is to identify craters in pictures of mars

import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
#from skimage import io, color
from skimage.feature import hog
from skimage import exposure
from sklearn.metrics import euclidean_distances
from sklearn.metrics import f1_score


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

################################################################################
###############              K Nearest Neighbours
###############################################################################3

#function to get features
def compute_hog(image):
    """ Compute HOG features and return them along with the visualized image. """
    features, hog_image = hog(image, 
                               orientations=4, 
                               pixels_per_cell=(24, 24), 
                               cells_per_block=(1, 1), 
                               visualize=True)
                               #multichannel=False)
    # Rescale histogram for better display
    #hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return features, hog_image


def distance_images(image1 , image2):
    #euclidean
    return np.sum(np.square(image1- image2) )

    #cosine
    #image1 = np.array(image1).reshape(48, 48)
    #image2 = np.array(image2).reshape(48, 48)
    #features1 ,hog_image1 = compute_hog(image1)
    #features2 ,hog_image2 = compute_hog(image2)
    #return cosine(features1, features2)



k_max = 37
y_val_pred = np.zeros( [k_max , len(X_val)] )
features_train = np.array( [compute_hog(np.array(X_train[i]).reshape(48, 48))[0] for i in range(len(X_train))] )
for i_val in range(len(X_val)): #chech all validation images
    features_i = compute_hog(np.array(X_val[i_val]).reshape(48, 48))[0]
    distances = [(distance_images(features_i, features_train) ,y_train[i]) for i in range(len(X_train))]
    distances.sort()
    if(i_val%10==0):
        print("Checking image = " + str(i_val))
    for k in range(1,k_max-1): #choose how many neighbours
        k_neighbours = distances[0:k]
        #voting = np.mean([t[1] for t in k_neighbours])
        voting  = np.sum( [t[0]*t[1] for t in k_neighbours ] )/np.sum([ t[0] for t in k_neighbours ])
        if voting < 0.5:
            voting = 0
        else:
            voting = 1
            y_val_pred[i_val][k] = voting
        
    
all_errors = [np.mean(abs(y_val - y_val_pred[k])) for k in range( k_max ) ]
all_f1     = [f1_score(y_val , y_val_pred[k]) for k in range( k_max ) ]



## plotting error with gradient steps  ##
plt.plot(all_errors, color='blue' , label = "Accuracy")
plt.plot(all_f1 , color= "red", label= "F1")
plt.xlabel('K neighbours used')
plt.ylabel('Number of errors(%)')
plt.title('Error percentage with k neighbours')
plt.show()



