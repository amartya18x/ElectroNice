from nicelectro import *
import theano
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor.nlinalg
from optimization import Adam
class NiceACDC(object):
    def __init__(self,input,n_in,num_layer):
        self.inp = input
        self.n_in = n_in
        ##Build the shared parameters
        self.diag_p = theano.shared(np.random.normal(1,0.01,size=(n_in,)))
        self.diag_sp = theano.shared(np.random.normal(0,0.01,size=(n_in,)))
        self.params =[]
        layers_enc = []


        coup = tuple((Params1(self.n_in)).params)
        self.params=(coup)
        layer1 = electronice(inp = input, n_in = n_in, A = coup[0], D = coup[1], Kx = coup[2], Ky = coup[3])
        layers_enc.append(layer1)

        
        for i in range(1,num_layer):
            coup =tuple((Params1(self.n_in)).params)
            self.params += (coup)
            layer = electronice(inp = layers_enc[i-1].output, n_in = n_in, A = coup[0], D = coup[1], Kx = coup[2], Ky = coup[3])
            layers_enc.append(layer)
            
        self.params += (self.diag_p,self.diag_sp)

        wrt = self.inp
        self.expr = (layers_enc[-1].output)*(self.diag_p).dimshuffle('x',0)+self.diag_sp.dimshuffle('x',0)
        newDet =  layers_enc[0].det
        for i in range(1,num_layer):
            newDet += layers_enc[i].det

        self.newLogDet = newDet + T.sum((T.log(abs(self.diag_p))))


        self.prior =T.mean(T.sum( -1* T.log(1+T.exp(self.expr)) - 1* T.log(1 + T.exp(-1*self.expr)),axis=1))

        self.cost = self.prior + self.newLogDet
        print type(self.params)
        print type(self.cost)
        print self.cost
        print self.params
        self.grad = T.grad(self.cost,self.params)
        self.updates_prior = Adam(self.prior,self.params)
        self.updates = Adam(self.cost,self.params)
class Sampler(object):
    def __init__(self,inp,n_in,params,num_layer):
        self.n_in = n_in
        diag_p = params[4*num_layer]
        diag_sp = params[4*num_layer + 1]
        self.inp = (inp-diag_sp)/(diag_p)
        self.layers_dec = []
        layer1 = electronice(inp = self.inp, n_in = n_in, A = params[4*num_layer-4], D = params[4*num_layer-3], Kx = params[4*num_layer-2], Ky = params[4*num_layer-1],inverse=True)
        self.layers_dec.append(layer1)

        
        for i in range(1,num_layer):
            layer = electronice(inp = self.layers_dec[i-1].output, n_in = n_in, A = params[4*(num_layer-i)-4], D = params[4*(num_layer-i)-3], Kx = params[4*(num_layer-i)-2], Ky = params[4*(num_layer-i)-1],inverse=True)
            self.layers_dec.append(layer)
            

        self.output = self.layers_dec[-1].output
        
if __name__ == '__main__':
    inp = T.vector("inp")
    n_in = 784
    num_layers=5
    ANice = NiceACDC(input = inp, n_in = n_in,num_layers=5)
    fn = theano.function([inp],[ANice.output])
    gn = theano.function([inp],ANice.grad)
    input  =np.random.uniform(size=(n_in,))
    print input
    #print gn(input)
    print fn(input)
