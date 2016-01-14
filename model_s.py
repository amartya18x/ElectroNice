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

        self.newLogDet = newDet


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
    def __init__(self,input,n_in,params):
        self.inp = input
        self.n_in = n_in
        
    #[A1_values,D1_values,Kx1,Ky1,A2_values,D2_values,Kx2,Ky2,A3_values,D3_values,Kx3,Ky3,A4_values,D4_values,Kx4,Ky4,A5_values,D5_values,Kx5,Ky5,diag_p,diag_sp] = params
    
        self.layer6_inp = (self.inp-diag_sp)/(diag_p)
        self.layer6 = electronice(
            n_in = n_in,
            inp = self.layer6_inp,
            A = A5_values,
            D = D5_values,
            Kx = Kx5,
            Ky = Ky5,
            inverse = True 
        )
        self.layer7 = electronice(
            n_in = n_in,
            inp = self.layer6.output,
            A = A4_values,
            D = D4_values,
            Kx = Kx4,
            Ky = Ky4,
            inverse = True
        )
        self.layer8 = electronice(
            n_in = n_in,
            inp = self.layer7.output,
            A = A3_values,
            D = D3_values,
            Kx = Kx3,
            Ky = Ky3,
            inverse = True
        )
        self.layer9 = electronice(
            n_in = n_in,
            inp = self.layer8.output,
            A = A2_values,D = D2_values,
            Kx = Kx2,
            Ky = Ky2,
            inverse = True
        )
        self.layer10 =  electronice(
            n_in = n_in,
            inp = self.layer9.output,
            A = A1_values,
            D = D1_values,
            Kx = Kx1,
            Ky = Ky1,
            inverse = True
        )
        self.output = self.layer10.output
        
if __name__ == '__main__':
    inp = T.vector("inp")
    n_in = 784
    ANice = NiceACDC(input = inp, n_in = n_in)
    fn = theano.function([inp],[ANice.output])
    gn = theano.function([inp],ANice.grad)
    input  =np.random.uniform(size=(n_in,))
    print input
    #print gn(input)
    print fn(input)
