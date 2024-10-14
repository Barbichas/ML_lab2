#Problemas Aprendizagem Automática
#2024_10_11

import numpy as np
import math




class one_layer_perceptron:

    def __init__(self):
        self.data = []
        self.labels = []
        self.w = []
        self.learning_step = 1
        self.steepness = 1 #inner steepness factor

    def weight_init(self ):
        self.w = np.ones( 1+self.data.shape[1])
        return

    def data_point_with_bias(self, x):
        return np.concatenate( ([1], x) )

    def sigmoid(self , net ):
        aux = math.exp(-self.steepness * net )
        return 1/(1 + aux)

    def prediction(self, x ):
        #print("weight vector = " +str(self.w) + " data point = " +str(x))
        return self.sigmoid( np.dot( self.w , x) )

    def Error(self):
        sum = 0
        for k in range(len(self.data)):
            pred = self.prediction( self.data_point_with_bias(self.data[k]) )
            sum += (self.labels[k] - pred)**2

        return sum

    def pred_labels(self):
        all = []
        for k in range(len(self.data)):
            all.append( self.prediction( self.data_point_with_bias(self.data[k]) ) )
        return all

    def pred_input(self, input ):
        all = []
        for j in range(len(input)):
            all.append( self.prediction( self.data_point_with_bias(self.data[k]) ) )
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
                o = self.prediction( self.data_point_with_bias(self.data[k]))
                if j == 0:
                    aux+= (t-o)*o*(1-o)*(-self.steepness)
                else:
                    aux += (t-o)*o*(1-o)*(-self.steepness*self.data[k][j-1])
            deltas.append(-1*self.learning_step*aux)
        #print("Weights are "+str(self.w))
        #print("Deltas are " + str(deltas))
        for j in range(len(self.w)):
            self.w[j]+=deltas[j]
        return deltas
            
    def stochastic_gradient_epoch(self):
        #print(self.data)
        for k in range(len(self.data)) : #for every point in dataset
            deltas = []
            t = self.labels[k]
            o = self.prediction( self.data_point_with_bias(self.data[k]) )
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


class two_layer_perceptron:

    np.random.seed(42)

    #horrible index explanation
    #k -> input data points
    #i -> inner layer sommas
    #j -> input signal features
    def __init__(self):
        self.data = []
        self.labels = []
        self.w = []
        self.W = []
        self.learning_step = 1
        self.steepness = [] #inner steepnesses, last one is output
        self.n_inner_sommas = 2

    def weight_init(self ):
        #self.W = np.random.rand( self.n_inner_sommas,1+self.data.shape[1] )
        #self.w = np.random.rand( 1 + self.n_inner_sommas)
        self.W = np.ones( [ self.n_inner_sommas,1+self.data.shape[1] ])
        self.w = np.ones( [ 1 + self.n_inner_sommas] )
        self.steepness = np.ones( 1 + self.n_inner_sommas) #last one is the output steepness
        return

    def data_point_with_bias(self, x):
        return np.concatenate( ([1], x) )

    def sigmoid(self , net , steep):
        aux = math.exp(-steep * net )
        return 1/(1 + aux)

    def prediction(self, x ):
        #print("weight vector = " +str(self.w) + " data point = " + str(x))
        V = []
        for i in range(self.n_inner_sommas):
            V.append( self.sigmoid( np.dot(self.W[i] , self.data_point_with_bias(x) ) , self.steepness[i] ) )
        #print("Shape of V = " + str(len(V)))
        return self.sigmoid( np.dot( self.w , self.data_point_with_bias(V)) , self.steepness[self.n_inner_sommas] )

    def Error(self):
        sum = 0
        for k in range(len(self.data)):
            pred = self.prediction( self.data[k] )
            sum += (self.labels[k] - pred)**2

        return sum

    def pred_labels(self):
        all = []
        for k in range(len(self.data)):
            all.append( self.prediction( self.data[k] ) )
        return all


    def gradient_step(self):
        
        #print("w  = "+str(self.w))
        #print("W0 = "+str(self.W[0]))
        #print("W1 = "+str(self.W[1]))

        deltas_k =[]  #backpropagated values directly from output
        V = np.zeros([len(self.data), self.n_inner_sommas+1] )
        deltas_ki = np.zeros([len(self.data), self.n_inner_sommas] ) #backpropagation for hidden layer

        for k in range(len(self.data)):
            xk = self.data_point_with_bias(self.data[k] )
            yk = self.labels[k]
            ok = self.prediction( self.data[k] )
            derivative_k = (self.steepness[self.n_inner_sommas]) * ok *(1-ok)
            current_delta_k = (yk-ok)* derivative_k 
            deltas_k.append( current_delta_k )
            for i in range(self.n_inner_sommas+1):
                if i == 0:
                    V[k][i] = 1
                else:
                    net_ki = np.dot( self.W[i-1] , xk )
                    V[k][i] = self.sigmoid(net_ki, self.steepness[i-1])
                    derivative_i = (self.steepness[i])*V[k][i]*(1-V[k][i])
                    deltas_ki[k][i-1] = current_delta_k * self.w[i-1] * derivative_i
                
        #print("Preditions = " +str( [self.prediction(self.data[k]) for k in range(len(self.data))] )  )
        #corrections on top layer weights
        for i in range( self.n_inner_sommas ):
            for k in range(len(self.data)):
                self.w[i]+=self.learning_step*deltas_k[k]*V[k][i]

        #corrections on hidden layer weights
        for i in range( self.n_inner_sommas ):
            for j in range( 1+self.data.shape[1] ):
                aux = 0
                for k in range( len(self.data) ):
                    if j == 0:
                        aux += self.learning_step * deltas_ki[k][i]
                    else:
                        aux += self.learning_step * deltas_ki[k][i]*self.data[k][j-1]
                self.W[i][j] += aux

        return


  









