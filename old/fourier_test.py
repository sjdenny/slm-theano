import numpy as np
import theano
import theano.tensor as T
from theano.gradient import DisconnectedType

import pdb

class GradTodo(theano.Op):
    def make_node(self, x):
        return theano.Apply(self, [x], [x.type()])
    def perform(self, node, inputs, outputs):
        raise NotImplementedError('TODO')

grad_todo = GradTodo()

class InverseFourierOp(theano.Op):
    __props__ = ()
    
    def make_node(self, xr, xi):
        # check that the theano version has support for __props__
        assert hasattr(self, '_props')
        xr = T.as_tensor_variable(xr)
        xi = T.as_tensor_variable(xi)
        
        return theano.Apply(self, [xr, xi], [xr.type(), xr.type()])
    
    def perform(self, node, inputs, output_storage):
        x = inputs[0] + 1j*inputs[1]
        nx, ny = inputs[0].shape
        z_r = output_storage[0]
        z_i = output_storage[1]
        s = np.fft.ifft2(x) * (nx*ny)
        z_r[0] = np.real(s)
        z_i[0] = np.imag(s)

xr = T.matrix('xr')
xi = T.matrix('xi')
ifft = InverseFourierOp()(xr, xi)
f_ifft = theano.function([xr, xi], ifft)

class FourierOp(theano.Op):
    __props__ = ()
    
    def make_node(self, xr, xi):
        # check that the theano version has support for __props__
        assert hasattr(self, '_props')
        xr = T.as_tensor_variable(xr)
        xi = T.as_tensor_variable(xi)
        
        return theano.Apply(self, [xr, xi], [xr.type(), xr.type()])
    
    def perform(self, node, inputs, output_storage):
        x = inputs[0] + 1j*inputs[1]
        z_r = output_storage[0]
        z_i = output_storage[1]
        s = np.fft.fft2(x)  # has "1" normalisation
        z_r[0] = np.real(s)
        z_i[0] = np.imag(s)
        
    def grad(self, inputs, output_gradients):
        """
        From the docs:
        If an Op has a single vector-valued output y and a single vector-valued input x,
        then the grad method will be passed x and a second vector z. Define J to be the 
        Jacobian of y with respect to x. The Op's grad method should return dot(J.T,z).
        When theano.tensor.grad calls the grad method, it will set z to be the gradient 
        of the cost C with respect to y. If this op is the only op that acts on x, then
        dot(J.T,z) is the gradient of C with respect to x. If there are other ops that 
        act on x, theano.tensor.grad will have to add up the terms of x's gradient 
        contributed by the other op's grad method.
        """        
        z_r = output_gradients[0]
        z_i = output_gradients[1]
        
        # check at least one is not disconnected:
        if (isinstance(z_r.type, DisconnectedType) and 
            isinstance(z_i.type, DisconnectedType)):
            return [DisconnectedType, DisconnectedType]
        
        if isinstance(z_r.type, DisconnectedType):
            print 'z_r using zeros_like'
            z_r = z_i.zeros_like()
        
        if isinstance(z_i.type, DisconnectedType):
            print 'z_i using zeros_like'
            z_i = z_r.zeros_like()
        
        y = InverseFourierOp()(z_r, z_i)
        return y



xr = T.matrix('xr')
xi = T.matrix('xi')
f_transform = FourierOp()(xr, xi)
f = theano.function([xr, xi], f_transform)
#inp = np.asarray(np.random.rand(4,4), dtype=theano.config.floatX)
N = 4

inp_r = np.random.rand(4,4)
inp_i = np.random.rand(4,4)
target = np.zeros_like(inp_r)
out = f(inp_r, inp_i)
print 'inp_r = \n{}\ninp_i = \n{}'.format(inp_r, inp_i)
print 'out = {}'.format(out)

# now make a very basic fourier transform test:
#print 'Gradients:'
f_transform_r, f_transform_i = FourierOp()(xr, xi)
cost = T.sum(T.pow(f_transform_r - target, 2))
f_cost = theano.function([xr, xi], cost)

cost_grad = T.grad(cost, wrt=[xr, xi])
f_cost_grad = theano.function([xr, xi], cost_grad)

out = f_cost_grad(inp_r, inp_i)

print 'cost_grad = \n{}'.format(out)

print ''
print 'Numerical gradient estimation:'
print '------------------------------'

eps = 1e-3
z_inp = np.zeros_like(inp_r)
grad_est_r = np.zeros_like(inp_r)
grad_est_i = np.zeros_like(inp_r)
for j in range(N):
    for k in range(N):
        inp_r[j,k] += eps
        cost_1 = f_cost(inp_r, inp_i)
        inp_r[j,k] -= 2*eps
        cost_2 = f_cost(inp_r, inp_i)
        inp_r[j,k] += eps
        grad_est_r[j,k] = (cost_1 - cost_2) / (2*eps)
        
for j in range(N):
    for k in range(N):
        inp_i[j,k] += eps
        cost_1 = f_cost(inp_r, inp_i)
        inp_i[j,k] -= 2*eps
        cost_2 = f_cost(inp_r, inp_i)
        inp_i[j,k] += eps
        grad_est_i[j,k] = (cost_1 - cost_2) / (2*eps)
        
print 'Est. gradient (r) = \n{}'.format(grad_est_r)
print 'Est. gradient (i) = \n{}'.format(grad_est_i)
print ''
print 'Verify ratios:'
print grad_est_r / out[0]
print grad_est_i / out[1]