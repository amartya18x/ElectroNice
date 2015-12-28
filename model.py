from nicelectro import *
import theano
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor.nlinalg

class NiceACDC(object):
    def __init__(self,input,n_in):
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
        self.params = coup1.params + coup2.params + coup3.params + coup4.params + coup5.params
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
        expr = self.layer5.output
        self.Jacob , updates = theano.scan(lambda i, y,x : theano.gradient.jacobian(expr[i], x), sequences=T.arange(expr.shape[0]), non_sequences=[expr,wrt]) 
        self.logDet  = theano.tensor.log(theano.tensor.nlinalg.Det()(self.Jacob))
        self.layer6 = electronice(
            n_in = n_in,
            inp = self.layer5.output,
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
            A = A2_values,
            D = D2_values,
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
        self.prior =T.sum( -T.log(1+T.exp(self.layer5.output)) - T.log(1 + T.exp(-1*self.layer5.output)))
        self.cost = self.prior + self.logDet
        self.grad = T.grad(self.cost,self.params)
inp = T.vector("inp")
n_in = 5
ANice = NiceACDC(input = inp, n_in = n_in)
fn = theano.function([inp],[ANice.output,ANice.prior,ANice.logDet])
gn = theano.function([inp],ANice.grad)
input  =np.random.uniform(size=(n_in,))
print input
#print gn(input)
print fn(input)
