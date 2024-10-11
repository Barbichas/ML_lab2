#Problemas Aprendizagem Automática
#2024_10_8

import numpy as np
import math

class perceptron:

    def __init__(self):
        self.data = []
        self.labels = []
        self.w = []
        self.learning_step = 1
        self.steepness = 1 #inner steepness factor

    def weight_init(self ):
        self.w = np.ones( 1+self.data.shape[1])
        return

    def sigmoid(self , net ):
        #print("net values this much "+str(net))
        aux = math.exp(-self.steepness * net )
        return 1/(1 + aux)

    def prediction(self, x ):
        return self.sigmoid( np.dot( self.w , x) )

    def Error(self):
        sum = 0
        for k in range(len(self.data)):
            pred = self.prediction(np.concatenate( (np.array([1]) , self.data[k]) ))
            sum += (self.labels[k] - pred)**2

        return sum

    def pred_labels(self):
        all = []
        for i in range(len(self.data)):

            all.append( self.prediction(np.concatenate( (np.array([1]) , self.data[i]) )) )
        return all

    def gradient_step(self):
        deltas = []
        #print(self.data)
        for j in range(len(self.w)):
            aux = 0
            #print("Checking weight " + str(j) )
            for k in range(len(self.data)) :
                #print("Datapoint " + str(k))
                t = self.labels[k]
                #print("Shape of current input is "+ str(self.data[k].shape))
                #print("Shape of current input with tweak is "+ str( ([1] + self.data[k]).shape ))
                #print(self.data[k])
                #print([1] + self.data[k])
                o = self.prediction(np.concatenate( (np.array([1]),self.data[k]) ) )
                #print("output is " +str(o))
                if j == 0:
                    aux+= (t-o)*o*(1-o)*(-self.steepness)
                else:
                    aux += (t-o)*o*(1-o)*(-self.steepness*self.data[k][j-1])
            if(aux != 0):
                print("aux sum is at " +str(aux))
            deltas.append(-1*self.learning_step*aux)
        #print("Weights are "+str(self.w))
        #print("Deltas are " + str(deltas))
        print("Sum of deltas is " +str(np.sum([abs(ii) for ii in deltas ])))
        for j in range(len(self.w)):
            self.w[j]+=deltas[j]
        return deltas

    def stochastic_gradient_epoch(self):
        #print(self.data)
        for k in range(len(self.data)) : #for every point in dataset
            deltas = []
            t = self.labels[k]
            o = self.prediction(np.concatenate( (np.array([1]),self.data[k]) ) )
            #print("t= "+str(t)+ " , o= "+str(o))
            aux = 0
            for j in range(len(self.w)): #for every weighted connection to output
                if j == 0:
                    aux = (t-o)*o*(1-o)*(-self.steepness)
                else:
                    aux = (t-o)*o*(1-o)*(-self.steepness*self.data[k][j-1])
                #if(aux != 0):
                #    print("aux(t-o) scaled is at " +str(aux))
                deltas.append(-1*self.learning_step*aux)
            #print("Sum of deltas is " +str(np.sum([abs(ii) for ii in deltas ])))
            for j in range(len(self.w)):
                self.w[j]+=deltas[j]
        return
            


    









