from nicelectro import *
import theano
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor.nlinalg
from optimization import Adam
class NiceACDC(object):
    def __init__(self,input,n_in):
        self.inp = input
        self.n_in = n_in
        ##Build the shared parameters
        coup1 = Params1(self.n_in)
        self.diag_p = theano.shared(np.random.normal(1,0.01,size=(n_in,)))
        self.diag_sp = theano.shared(np.random.normal(0,0.01,size=(n_in,)))
        [A1_values,D1_values,Kx1,Ky1] = coup1.params
        coup2 = Params1(self.n_in)
        [A2_values,D2_values,Kx2,Ky2] = coup2.params
        coup3 = Params1(self.n_in)
        [A3_values,D3_values,Kx3,Ky3] = coup3.params
        coup4 = Params1(self.n_in)
        [A4_values,D4_values,Kx4,Ky4] = coup4.params
        coup5 = Params1(self.n_in)
        [A5_values,D5_values,Kx5,Ky5] = coup5.params
        self.params = coup1.params + coup2.params + coup3.params + coup4.params + coup5.params + [self.diag_p,self.diag_sp]
        #self.params = coup1.params + coup2.params + coup3.params
        ##Build the layers
        self.layer1 = electronice(
            inp = input,
            n_in = n_in,
            A = A1_values,
            D = D1_values,
            Kx = Kx1,
            Ky = Ky1
        )
        self.layer2 = electronice(
            n_in = n_in,
            inp = self.layer1.output,
            A = A2_values,
            D = D2_values,
            Kx = Kx2,
            Ky = Ky2
        )
        self.layer3 = electronice(
            n_in = n_in,
            inp = self.layer2.output,
            A = A3_values,
            D = D3_values,
            Kx = Kx3,
            Ky = Ky3
        )
        self.layer4 = electronice(
            n_in = n_in,
            inp = self.layer3.output,
            A = A4_values,
            D = D4_values,
            Kx = Kx4,
            Ky = Ky4
        )
        self.layer5 = electronice(
            n_in = n_in,
            inp = self.layer4.output,
            A = A5_values,
            D = D5_values,
            Kx = Kx5,
            Ky = Ky5
        )
        wrt = self.inp
        self.expr = (self.layer5.output)*(self.diag_p).dimshuffle('x',0)+self.diag_sp.dimshuffle('x',0)
        self.newDet = self.layer1.det+self.layer2.det+self.layer3.det+self.layer4.det+self.layer5.det+T.sum((T.log(abs(self.diag_p))))
        self.newLogDet = self.newDet
        layer6_inp = (self.inp-self.diag_sp.dimshuffle('x',0))/(self.diag_p).dimshuffle('x',0)
        self.layer6 = electronice(
            n_in = n_in,
            inp = layer6_inp,
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
        self.prior =T.mean(T.sum( -1* T.log(1+T.exp(self.expr)) - 1* T.log(1 + T.exp(-1*self.expr)),axis=1))
        #self.prior = T.mean(T.log(T.prod(T.exp(-self.layer5.output*self.layer5.output)/np.sqrt(2*np.pi),axis=1)))
        self.cost = self.prior + self.newLogDet
        self.grad = T.grad(self.cost,self.params)
        self.updates_prior = Adam(self.prior,self.params)
        self.updates = Adam(self.cost,self.params)
class Sampler(object):
    def __init__(self,input,n_in,params):
        self.inp = input
        self.n_in = n_in
        
        [A1_values,D1_values,Kx1,Ky1,A2_values,D2_values,Kx2,Ky2,A3_values,D3_values,Kx3,Ky3,A4_values,D4_values,Kx4,Ky4,A5_values,D5_values,Kx5,Ky5,diag_p,diag_sp] = params
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
