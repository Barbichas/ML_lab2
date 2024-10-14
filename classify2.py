#13.10.2024
#Machine Learning lab
#The goal is to identify craters in pictures of mars

import numpy as np
import matplotlib.pyplot as plt
import two_layer_perceptron_Carlos as Perceptron

X_train = np.load("Xtrain1.npy")
y_train = np.load("Ytrain1.npy")

########    Counting labels    ##########
train_total = y_train.shape[0]
n_craters = 0
n_plain   = 0
for element in y_train:
    if element== 0:
        n_plain += 1
    else: n_craters += 1

print("DESCRIPTION OF DATASET")
print()
print("Number of training images = " + str(train_total))
print("Percentage of craters is " + str(100*n_craters/train_total) + " %")
print("Percentage of plain is " + str(100*n_plain/train_total) + " %")


########   Shortening for now  ##########
print("Shape of input data is " + str(X_train.shape))
X_train = X_train[0:10]
y_train = y_train[0:10]

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

if(0):
    for i in range(20):
        display_image(X_train[i],48,48)
        
        if(Y_train[i]==1):
            plt.title("Image "+ str(i) + " has a cratter")
        else:
            plt.title("Image "+ str(i) + " is clean")
        plt.figure()

P = Perceptron.two_layer_perceptron()
P.data = X_train_normalised
P.labels = y_train_normalised
P.learning_step = 0.5
P.n_inner_sommas = 37
P.weight_init()



#print(np.array(y_train))
print("----------------------------------------")
n_epochs = 200
all_errors =[]
for i in range(n_epochs):
    #print("----------------------------------")
    #print("Updating with deltas = "+ str(P.gradient_step() ))
    pred_labels = P.pred_labels()
    #pred_labels = [round(ii) for ii in pred_labels]
    error = np.sum(np.square(pred_labels-y_train))
    all_errors.append(error)

    P.gradient_step()
    
    if(i%10==0):
        print("Iteration "+ str(i))
        #print(pred_labels)
        #print("Predicted-label error =" + str(y_train-pred_labels) + "Error =" + str(error))

    #print( "Current error= " + str(P.Error() ))

## plotting error with gradient steps  ##
plt.plot(all_errors, color='blue')
plt.xlabel('Iteration')
plt.ylabel('Squared errors')
plt.title('Error evolution while learning')
plt.show()


