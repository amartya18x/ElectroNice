import theano
import numpy as np
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
class electronice(object):
    def __init__(self,  inp, n_in, A,D,Kx,Ky,inverse=False):
        '''
        inp : Input tensor
        n_in : Dimension of input tensor
        A,D : Model parameters for that layer
        Kx,Ky : Piecewise-linerity parameters
        inverse : True if this layer is a part of the decoding network
        '''
        activation = self.knot_activ
        self.inp = inp

        # Compute C values. Need to improve this to O(nlog(n)) implementation
        C_values = self.dct_matrix(n_in,n_in).astype(theano.config.floatX)
        C = theano.shared(value=C_values,name='C',borrow=True)
        C_Tvalues = self.dct_matrix(n_in,n_in).astype(theano.config.floatX)
        C_t = theano.shared(value=C_values.T,name='C_t',borrow=True)   
        
        self.Kx = Kx
        self.Ky = Ky
        self.A = A
        self.D = D
        if inverse == False:
            #If part of the encoding lauyer
            self.h1 = inp*(T.tanh(self.A)).dimshuffle('x',0) #multiply by diagonal A
            self.h2 = T.dot(self.h1,C) #DCT transform
            self.h3 = self.h2*(self.D).dimshuffle('x',0) #multiply by diagonal D
            self.lin_output = T.dot(self.h3,C_t) #Inverse DCT transform
            self.output = self.knot_activ(self.lin_output) #piecewise linear activation
            self.activD = self.knot_Det(self.lin_output) # calculate the diagonal elements of the jacobian of the non-linearity
            self.det = T.sum(T.log(abs(T.tanh(self.A))))+(T.sum(T.log(abs(self.D))))+T.mean(T.log(abs(T.prod(self.activD,axis=0))))# Calculate the contribution of the layer towards the log-det term
        else:
            #Same as the encoding part with minor mods
            self.h1 = self.knot_activ_inverse(inp)
            self.h2 = T.dot(self.h1,C)
            self.h3 = self.h2* (1/self.D).dimshuffle('x',0)
            self.h4 = T.dot(self.h3,C_t)
            self.lin_output = self.h4*(1/T.tanh(self.A)).dimshuffle('x',0)
            self.output = self.lin_output
            self.activD = self.knot_Det_inverse(inp)
            self.det = T.sum(T.log(abs(T.tanh(self.A))))+(T.sum(T.log(abs(self.D))))+T.mean(T.log(abs(T.prod(self.activD,axis=0))))
        self.params = [self.A, self.D, self.Kx,self.Ky]
        
    def knot_activ(self,x):
        '''
        Piecewise linear activation
        If x < tanh(Kx):
             y = (x+1)*(tanh(Ky)+1)/(tanh(Kx) + 1) - 1
        Else
             y = (x-1)*(tanh(Ky) - 1)/(tanh(Kx) - 1) + 1
        '''
        return (x<T.tanh(self.Kx))*( (x+1)*(T.tanh(self.Ky)+1)/(T.tanh(self.Kx)+1) - 1 )  + (x>=T.tanh(self.Kx)) *( (x-1)*(T.tanh(self.Ky)-1)/(T.tanh(self.Kx)-1) + 1 )

    def knot_activ_inverse(self,x):
        '''
        Calculate the inverse of the piee-wise linearity
        It is the inverse of the previous function
        '''
        return (x<T.tanh(self.Ky))*( (x+1)*(T.tanh(self.Kx)+1)/(T.tanh(self.Ky)+1) - 1 )  + (x>=T.tanh(self.Ky)) *( (x-1)*(T.tanh(self.Kx)-1)/(T.tanh(self.Ky)-1) + 1 )
    
    def knot_Det(self,x):
        '''
        Calculate the Jacobian of the piece-wise linear activation
        '''
        return (x<T.tanh(self.Kx))*( (T.tanh(self.Ky)+1)/(T.tanh(self.Kx)+1)  )  + (x>=T.tanh(self.Kx)) *( (T.tanh(self.Ky)-1)/(T.tanh(self.Kx)-1)  )

    def knot_Det_inverse(self,x):
        '''
        Returns the Jacobian of the piece-wise linear activation for the inverse transformation
        '''
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
    '''
    Generate Parameters
    '''
    def __init__(self,n_in,A=None,D=None,Kx=None,Ky=None,Db=None):
        #Db is useless
        if A is None:
            #Generate one centred gaussian noise with 0.1 variance as the diagnoal layer A
            A_values = np.random.normal(1,0.1,(n_in,))
        else:
            A_values = A
        A = theano.shared(A_values,borrow=True)
        
        if D is None:
            #Generate one centred gaussian noise with 0.1 variance as the diagnoal layer A
            D_values = np.random.normal(1,0.1,(n_in,))
        else:
            D_values = D
        D = theano.shared(D_values, borrow=True)

        if Kx is None:
            #Kx is the projection of the knot on the x-axis
            Kx = np.random.normal(1,0.1,(n_in,))
            
        if Db is None:#THis is useless
            Db_values = np.random.normal(0,0.1,(n_in,))
        else:
            Db_values = Db
        
        if Ky is None:
            #Ky is the projection of the knot on the y-axis
            Ky = np.random.normal(1,0.1,(n_in,))

            
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
