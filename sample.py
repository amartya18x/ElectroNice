import theano
import theano.tensor as T
import numpy as np
from model_s import Sampler
import cPickle
import gzip
import os
import sys
import timeit
from datetime import datetime
inp = T.matrix("inp")
n_in = 784
num_layer = 25
theano.config.floatX = 'float32'

f = open('best_model.pkl')
param = cPickle.load(f)
f.close()

#Generate the sampler model

sampler = Sampler(inp,n_in,param,num_layer=25)




#Sample function
sample_model = theano.function(
        inputs=[inp],
        outputs=[sampler.output]
        )


import matplotlib.pyplot as plt
#%matplotlib inline
np.random.seed(datetime.now().time().microsecond)

#Generate the logistic random sample

samp = np.random.logistic(loc = 0.0 , scale = .1, size=(1,784)).astype('float32')








#samples = np.random.uniform(size=(1,784)).astype('float32')
#samp =  np.log(samples) - np.log(1-samples)

#Reducing the variance in the data(From the observation)
gene = np.asarray(sample_model(samp/1)).reshape(28,28)
#gene[gene<-0.5]=-1

#We had normalized previously and so bringing it back
image = (0.5*(gene*748)+1)*255

#image[image<200] = 0
#image[image>200] = 255
#print (0.5*(gene*748)+1)*255
print image
plt.imshow(image,cmap='Greys')
plt.show()
#samp = np.random.logistic(loc=0,scale=1,size=(1,784)).astype('float32')
#samples = np.random.uniform(size=(1,784)).astype('float32')
#samp =  np.log(samples) - np.log(1-samples)
#gene = np.asarray(sample_model(samp/1)).reshape(28,28)
#print gene
#print ((((gene*784)+1)*0.5)*255.0)

#plt.imshow(0.5*((gene*748)+1)*255/np.sum(((gene*784)+1)*0.5),cmap='Greys')
#plt.show()
