import theano
import numpy as np
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
class electronice(object):
    def __init__(self,  inp, n_in, A,D,Kx,Ky,inverse=False):
        activation = self.knot_activ
        self.inp = inp
        
        C_values = self.dct_matrix(n_in,n_in).astype(theano.config.floatX)
        C = theano.shared(value=C_values,name='C',borrow=True)
        C_Tvalues = self.dct_matrix(n_in,n_in).astype(theano.config.floatX)
        C_t = theano.shared(value=C_values.T,name='C_t',borrow=True)   

        self.Kx = Kx
        self.Ky = Ky
        self.A = A
        self.D = D
        if inverse == False:
            self.h1 = T.dot(inp, T.diag(T.tanh(self.A))) 
            self.h2 = T.dot(self.h1,C)
            self.h3 = T.dot(self.h2,T.diag(T.tanh(self.D)))
            self.lin_output = T.dot(self.h3,C_t)
            #self.output,_ = theano.scan(lambda i,list_x : self.knot_activ(list_x[i]), sequences=T.arange(self.lin_output.shape[0]), non_sequences=self.lin_output)
            self.output = self.knot_activ(self.lin_output)
            self.activD,_ = theano.scan(lambda i,list_x : self.knot_Det(list_x[i]), sequences=T.arange(self.lin_output.shape[0]), non_sequences=self.lin_output)
            self.det = T.log(T.prod(T.tanh(self.A))*T.prod(T.tanh(self.D))*T.prod(self.activD))
        else:
            #self.h1,_ = theano.scan(lambda i,list_x : self.knot_activ_inverse(list_x[i]), sequences=T.arange(inp.shape[0]), non_sequences=inp)
            self.h1 = self.knot_activ_inverse(inp)
            self.h2 = T.dot(self.h1,C)
            self.h3 = T.dot(self.h2, T.diag(1/T.tanh(self.D))) 
            self.h4 = T.dot(self.h3,C_t)
            self.lin_output = T.dot(self.h4,T.diag(1/T.tanh(self.A)))
            self.output = self.lin_output
            self.activD,_ = theano.scan(lambda i,list_x : self.knot_Det_inverse(list_x[i]), sequences=T.arange(self.inp.shape[0]), non_sequences=inp)
            self.det = T.log(T.prod(T.tanh(self.A))*T.prod(T.tanh(self.D))*T.prod(self.activD))
        self.params = [self.A, self.D, self.Kx,self.Ky]
        
    def knot_activ(self,x):
        return (x<T.tanh(self.Kx))*( (x+1)*(T.tanh(self.Ky)+1)/(T.tanh(self.Kx)+1) - 1 )  + (x>=T.tanh(self.Kx)) *( (x-1)*(T.tanh(self.Ky)-1)/(T.tanh(self.Kx)-1) + 1 )

    def knot_activ_inverse(self,x):
        return (x<T.tanh(self.Ky))*( (x+1)*(T.tanh(self.Kx)+1)/(T.tanh(self.Ky)+1) - 1 )  + (x>=T.tanh(self.Ky)) *( (x-1)*(T.tanh(self.Kx)-1)/(T.tanh(self.Ky)-1) + 1 )
    
    def knot_Det(self,x):
        return (x<T.tanh(self.Kx))*( (T.tanh(self.Ky)+1)/(T.tanh(self.Kx)+1)  )  + (x>=T.tanh(self.Kx)) *( (T.tanh(self.Ky)-1)/(T.tanh(self.Kx)-1)  )

    def knot_Det_inverse(self,x):
        return (x<T.tanh(self.Ky))*( (T.tanh(self.Kx)+1)/(T.tanh(self.Ky)+1)  )  + (x>=T.tanh(self.Ky)) *((T.tanh(self.Kx)-1)/(T.tanh(self.Ky)-1)  )
    def dct_matrix(self,rows, cols, unitary=True):
        rval = np.zeros((rows, cols))
        col_range = np.arange(cols)
        scale = np.sqrt(2.0 / cols)
        for i in xrange(rows):
            rval[i] = np.cos(i * (col_range * 2 + 1) / (2.0 * cols) * np.pi) * scale
        if unitary:
            rval[0] *= np.sqrt(0.5)
        return rval


class Params1(object):
    def __init__(self,n_in,A=None,D=None,Kx=None,Ky=None):
        if A is None:
            A_values = np.random.normal(1,0.1,(n_in,))
        else:
            A_values = A
        A = theano.shared(A_values,borrow=True)
        
        if D is None:
            D_values = np.random.normal(1,0.1,(n_in,))
        else:
            D_values = D
        D = theano.shared(D_values, borrow=True)

        if Kx is None:
            Kx = 0.0
        
        if Ky is None:
            Ky = 0.0

        Kx = theano.shared(Kx)
        Ky = theano.shared(Ky)
        self.params = [A,D,Kx,Ky]



if __name__ == '__main__':        
    inp = T.vector('inp')
    srng = RandomStreams(seed=1235)
    n_in = 5
    #A_values = np.random.normal(0,0.02,(n_in,))
    #D_values = np.random.normal(0,0.02,(n_in,))
    coup1 = Params1(n_in)
    [A_values,D_values,Kx,Ky] = coup1.params
    layer1 = electronice(inp,n_in,A=A_values,D=D_values,Kx=Kx,Ky=Ky)
    layer2 = electronice(n_in=n_in,inp=layer1.output,A=A_values,D=D_values,Kx=Kx,Ky=Ky ,inverse=True)
    cost = T.sum(layer2.output)
    params = layer1.params + layer2.params 
    grad = T.grad(cost,params)
    fn = theano.function([inp],[layer2.output])
    gn = theano.function([inp],grad)
    input  =np.random.uniform(size=(n_in,))
    print input
    #print gn(input)
    print fn(input)
