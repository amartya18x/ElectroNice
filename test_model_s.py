from nicelectro import *
import theano
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor.nlinalg
class NiceACDC(object):
    def __init__(self,input,n_in,params,num_layer):
        self.inp = input
        self.n_in = n_in
        self.num_layer = num_layer
        self.diag_p = params[4*num_layer]
        self.diag_sp = params[4*num_layer + 1]

        self.layers_enc=[]
        layer_enc1 = electronice(inp = self.inp, n_in = n_in, A = params[0], D = params[1], Kx = params[2], Ky = params[3])
        self.layers_enc.append(layer_enc1)
        for i in range(1,num_layer):
            layer = electronice(inp = self.layers_enc[i-1].output, n_in = n_in, A = params[4*i], D = params[4*i+1], Kx = params[4*i+2], Ky = params[4*i+3],)
            self.layers_enc.append(layer)

        self.expr = (self.layers_enc[-1].output)*(self.diag_p).dimshuffle('x',0)+self.diag_sp.dimshuffle('x',0)
        newDet =  self.layers_enc[0].det
        for i in range(1,num_layer):
            newDet += self.layers_enc[i].det

        self.newLogDet = newDet + T.sum((T.log(abs(self.diag_p))))


        self.prior =T.mean(T.sum( -1* T.log(1+T.exp(self.expr)) - 1* T.log(1 + T.exp(-1*self.expr)),axis=1))

        self.cost = self.prior + self.newLogDet

        
        self.dec_inp = (self.expr -self.diag_sp)/self.diag_p
        self.layers_dec = []
        layer_dec1 = electronice(inp = self.dec_inp, n_in = n_in, A = params[4*num_layer-4], D = params[4*num_layer-3], Kx = params[4*num_layer-2], Ky = params[4*num_layer-1],inverse=True)
        self.layers_dec.append(layer_dec1)

        
        for i in range(1,num_layer):
            layer = electronice(inp = self.layers_dec[i-1].output, n_in = n_in, A = params[4*(num_layer-i)-4], D = params[4*(num_layer-i)-3], Kx = params[4*(num_layer-i)-2], Ky = params[4*(num_layer-i)-1],inverse=True)
            self.layers_dec.append(layer)
            

        self.output = self.layers_dec[-1].output
        self.reconstruction_err = T.sum(T.nnet.binary_crossentropy(self.output,input))
