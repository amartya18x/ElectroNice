import theano
import theano.tensor as T
import numpy as np
from model import NiceACDC
import cPickle
import gzip
import os
import sys
import timeit
inp = T.matrix("inp")
n_in = 784
lr = theano.shared(0.0001)
theano.config.floatX = 'float32'

dataset='mnist.pkl.gz'
f = gzip.open(dataset, 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
print '... building the model'
x = T.matrix('x')
ANice = NiceACDC(input = x, n_in = n_in)
print "Created model"
llh = ANice.cost
output = ANice.output
update = ANice.updates
print "Obtained Grads"
print "Created updates"

train_model = theano.function(
        inputs=[x],
        outputs=[ANice.newLogDet,ANice.prior,ANice.cost],
    updates=update
        )
train_set_x, train_set_y = train_set
p = len(train_set_x)
batch_size = 100
n_batches = p/batch_size
s = -1
llh = -np.inf
while 1:
    s += 1
    batch_index = s%(n_batches-1)
    a= train_set_x[batch_index*batch_size:(batch_index+1)*batch_size]
    b = (a/784)
    cost =  train_model(b)
    llh = cost[2]
    if s%50 == 0:
        print cost
        with open('best_model.pkl', 'w') as f:
                        cPickle.dump(ANice.params, f)
