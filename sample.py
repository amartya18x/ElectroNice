import theano
import theano.tensor as T
import numpy as np
from model import Sampler
import cPickle
import gzip
import os
import sys
import timeit
inp = T.matrix("inp")
n_in = 784
theano.config.floatX = 'float32'

f = open('best_model.pkl')
param = cPickle.load(f)
f.close()
sampler = Sampler(inp,n_in,param)
sample_model = theano.function(
        inputs=[inp],
        outputs=[sampler.layer10.output]
        )
import matplotlib.pyplot as plt
#%matplotlib inline
samp = np.random.logistic(size=(1,784)).astype('float32')
gene = np.asarray(sample_model(samp)).reshape(28,28)
plt.imshow(gene*255*748,cmap='Greys')
plt.show()