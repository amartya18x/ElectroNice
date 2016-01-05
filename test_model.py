from nicelectro import *
import theano
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor.nlinalg
class NiceACDC(object):
    def __init__(self,input,n_in,params):
        self.inp = input
        self.n_in = n_in
        [A1_values,D1_values,Kx1,Ky1,A2_values,D2_values,Kx2,Ky2,A3_values,D3_values,Kx3,Ky3,A4_values,D4_values,Kx4,Ky4,A5_values,D5_values,Kx5,Ky5,diag_p] = params
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
        self.expr = self.layer5.output*(diag_p).dimshuffle('x',0)
        self.newDet = self.layer1.det+self.layer2.det+self.layer3.det+self.layer4.det+self.layer5.det+T.sum((T.log(abs(diag_p))))
        self.newLogDet = self.newDet
        self.layer6_inp = self.expr/(diag_p).dimshuffle('x',0)
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
        self.prior =T.mean(T.sum( -T.log(1+T.exp(self.expr)) - T.log(1 + T.exp(-1*self.expr)),axis=1))
        self.cost = self.prior + self.newLogDet
        self.reconstruction_err = T.sum(T.nnet.binary_crossentropy(self.output,input))
