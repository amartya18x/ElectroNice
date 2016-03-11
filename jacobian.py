import theano
import theano.tensor as T
import numpy as np

def jacobian(expr, wrt):
    J, updates = theano.scan(lambda i, y,x : theano.gradient.jacobian(expr[i], x), sequences=T.arange(expr.shape[0]), non_sequences=[expr,wrt])
    return J
def dct_matrix(rows, cols, unitary=True):
    rval = np.zeros((rows, cols))
    col_range = np.arange(cols)
    scale = np.sqrt(2.0 / cols)
    for i in xrange(rows):
        rval[i] = np.cos(i * (col_range * 2 + 1) / (2.0 * cols) * np.pi) * scale
    if unitary:
        rval[0] *= np.sqrt(0.5)
    return rval

if __name__ == '__main__':
    x1 = T.dvector('x1')
    z = np.asarray(5)
    flag = T.le(z,x1[0])
    Y= T.concatenate([(flag*(x1[0]*2+3*x1[1])+(1-flag)*(x1[0]*4+x1[1]*5) ).dimshuffle('x'), (flag*(x1[0]*9+8*x1[1])+(1-flag)*(x1[0]*7+x1[1]*6) ).dimshuffle('x')])
    J = jacobian(Y,x1)
    fn = theano.function([x1],J)
    print fn([1,2])
    C = dct_matrix(8,8)
    Ct = C.T
    print C
    print Ct
    I = np.dot(C,Ct)
    I[I<10e-10] = 0
    print I
