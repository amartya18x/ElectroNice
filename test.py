import theano
import theano.tensor as T
import numpy as np
from test_model_s import NiceACDC
import cPickle
import gzip
import os
import sys
import timeit
import matplotlib.pyplot as plt
inp = T.matrix("inp")
n_in = 784
lr = theano.shared(0.0001)
theano.config.floatX = 'float32'

dataset='mnist.pkl.gz'
f = gzip.open(dataset, 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

f = open('best_model.pkl')
param = cPickle.load(f)
f.close()
print '... building the model'
x = T.matrix('x')
ANice = NiceACDC(input = x, n_in = n_in,params=param,num_layer=5)
print "Created model"

test_model = theano.function(
        inputs=[x],
        outputs=[ANice.newLogDet,ANice.prior,ANice.cost,ANice.expr,ANice.reconstruction_err,ANice.output]
        )
print "built function"
test_set_x, test_set_y = test_set
p = len(test_set_x)
batch_size = 1
n_batches = p/batch_size
s = -1
llh = -np.inf
#for i in range(0,n_batches-1):
for i in range(1,2):
    a= test_set_x[i:i+1]
    plt.imshow(a.reshape(28,28)*255,cmap='Greys')
    plt.show()
    a = 2*a - 1
    b = (a/784)
    cost =  test_model(b)
    #print cost[3]
    gene = cost[5].reshape(28,28)
    plt.imshow(0.5*((gene*748)+1)*255,cmap='Greys')
    plt.show()
print cost

