import theano
import theano.tensor as T
import numpy as np
from model_s import Sampler
import cPickle
import gzip
import os
import sys
import timeit
inp = T.matrix("inp")
n_in = 784
num_layer = 5
theano.config.floatX = 'float32'

f = open('best_model.pkl')
param = cPickle.load(f)
f.close()
sampler = Sampler(inp,n_in,param,num_layer=30)
sample_model = theano.function(
        inputs=[inp],
        outputs=[sampler.output]
        )
import matplotlib.pyplot as plt
#%matplotlib inline
#samp = np.random.logistic(size=(1,784)).astype('float32')
samples = np.random.uniform(size=(1,784)).astype('float32')
samp =  np.log(samples) - np.log(1-samples)
gene = np.asarray(sample_model(samp)).reshape(28,28)
print gene
print ((gene*748)+1)*0.5

plt.imshow(0.5*((gene*748)+1),cmap='Greys')
plt.show()
