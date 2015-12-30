import theano
import theano.tensor as T
import numpy as np
from model import NiceACDC
import cPickle
import gzip
import os
import sys
import timeit
inp = T.vector("inp")
n_in = 784
lr = theano.shared(0.0001)
theano.config.floatX = 'float32'



# In[11]:



dataset='mnist.pkl.gz'
f = gzip.open(dataset, 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()


# In[12]:

learning_rate=0.13
n_epochs=1000
dataset='mnist.pkl.gz'


######################
# BUILD ACTUAL MODEL #
######################
print '... building the model'

# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch

# generate symbolic variables for input (x and y represent a
# minibatch)
x = T.vector('x')  
# construct the logistic regression class
# Each MNIST image has size 28*28
ANice = NiceACDC(input = x, n_in = n_in)
print "Created model"
llh = ANice.cost
output = ANice.output
NiceParam  = ANice.params
grads = ANice.grad
update = []
print "Obtained Grads"
for param,grad in zip(NiceParam,grads):
    update.append((param,param+grad*lr))

print "Created updates"


# In[13]:

train_model = theano.function(
        inputs=[x],
        outputs=[ANice.inp,ANice.cost],
    updates=update
        )


# In[18]:

train_set_x, train_set_y = train_set
a= train_set_x[1]
b = (a*2)-1


# In[ ]:

for s in range(1,200000):
    d,e = train_model(a)
    if s%20 == 0:
        print e
