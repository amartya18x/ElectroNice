import theano
import theano.tensor as T
import numpy as np
from model_s import NiceACDC
import cPickle
import gzip
import os
import sys
import timeit

#Input matrix
inp = T.matrix("inp")
#Dimension of data point
n_in = 784
lr = theano.shared(0.0001)
theano.config.floatX = 'float32'


#Get the dataset
dataset='mnist.pkl.gz'
f = gzip.open(dataset, 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()



print '... building the model'

#Create the model
x = T.matrix('x')
ANice = NiceACDC(input = x, n_in = n_in,num_layer=5)
print "Created model"

#THis is the total log-likelihood
llh = ANice.cost

#This is the ADAM update rule
update = ANice.updates

print "Created updates"


#Training function
'''
newLogDet is the log determinant of the jacobian term
prior is the logistic loss term
cost is the sum of these two
'''
train_model = theano.function(
        inputs=[x],
        outputs=[ANice.newLogDet,ANice.prior,ANice.cost],
    updates=update
        )


#Train Set
train_set_x, train_set_y = train_set
p = len(train_set_x)

#Batch Size
batch_size = 500

#Number of batches
n_batches = p/batch_size
#n_batches = 2
s = -1
llh = -np.inf
while 1:
    s += 1
    #Get the new batch number
    batch_index = s%(n_batches-1)
    #Get the entire batch of datapoints
    a= train_set_x[batch_index*batch_size:(batch_index+1)*batch_size]

    #Normalize to make them lie between -1 and +1 whereas the intial lie between 0 and 1
    a = 2*a - 1

    #Ensuring their norm is less than 1 as |x_i| \forall i \in {0, n_in} \le 1
    b = (a/784)
    #Get the cost
    cost = train_model(b)
    llh = cost[2]
    if s%100 == 0:
        print cost
        with open('best_model.pkl', 'w') as f:
                        cPickle.dump(ANice.params, f)
