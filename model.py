from nicelectro import *
import theano
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams

class NiceACDC(object):
    def __init__(self,rng,input,n_in):
        self.rng = rng
        self.inp = input
        self.n_in = n_in
        ##Build the shared parameters
        coup1 = Params1(self.n_in)
        [A1_values,D1_values,Kx1,Ky1] = coup1.params
        coup2 = Params1(self.n_in)
        [A2_values,D2_values,Kx2,Ky2] = coup2.params
        coup3 = Params1(self.n_in)
        [A3_values,D3_values,Kx3,Ky3] = coup3.params
        coup4 = Params1(self.n_in)
        [A4_values,D4_values,Kx4,Ky4] = coup4.params
        coup5 = Params1(self.n_in)
        [A5_values,D5_values,Kx5,Ky5] = coup5.params
        ##Build the layers
        self.layer1 = electronice(
            srng,inp,n_in,A=A_values,D=D_values,Kx=Kx,Ky=Ky)
        layer2 = electronice(rng=srng,n_in=n_in,inp=layer1.output,A=A_values,D=D_values,Kx=Kx,Ky=Ky ,inverse=True)
        cost = T.sum(layer2.output)
        params = layer1.params + layer2.params 
        grad = T.grad(cost,params)
        fn = theano.function([inp],[layer1.output])
        gn = theano.function([inp],grad)
        input  =np.random.uniform(size=(n_in,))
        print input
        print gn(input)
        print fn(input)

