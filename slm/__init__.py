import numpy as np
import theano
import theano.tensor as T
from theano.gradient import DisconnectedType

class slmOptimisation(object):
    
    def __init__(self, target, initial_phi, profile_s=None, A0=1.0):
        self.target = target
        self.n_pixels = target.shape[0] / 2   # target should be 512x512, but SLM pattern calculated should be 256x256.
        self.intensity_calc = None
        
        if profile_s is None:
            profile_s = np.ones((self.n_pixels, self.n_pixels))
            
        self.profile_s = profile_s.astype(theano.config.floatX)
        self.A0 = A0
        
        # Set up cost function:
        self.phi    = theano.shared(value=initial_phi.astype(theano.config.floatX),
                                    name='phi')
        self.phi_rate = theano.shared(value=np.zeros_like(initial_phi).astype(theano.config.floatX),
                                      name='phi_rate')
        self.S_i = T.matrix('S_i')
        self.S_r = T.matrix('S_r')
        
        # E_in: (n_pixels**2)
        self.E_in_r = self.A0 * (self.S_r*T.cos(self.phi) - self.S_i*T.sin(self.phi))
        self.E_in_i = self.A0 * (self.S_i*T.cos(self.phi) + self.S_r*T.sin(self.phi))
        
        # E_in padded: (4n_pixels**2)
        self.zero_matrix = T.matrix('zero')
        idx_0, idx_1 = get_centre_range(self.n_pixels)
        self.E_in_r_pad = T.set_subtensor(self.zero_matrix[idx_0:idx_1,idx_0:idx_1], self.E_in_r)
        self.E_in_i_pad = T.set_subtensor(self.zero_matrix[idx_0:idx_1,idx_0:idx_1], self.E_in_i)
        
        # E_out:
        # f = theano.function([xr, xi], FourierOp()(xr, xi))
        self.E_out_r, self.E_out_i = fft(self.E_in_r_pad, self.E_in_i_pad)        
        
        # finally, the output intensity:
        #self.E_out_2 = T.pow(self.E_out_r, 2) + T.pow(self.E_out_i, 2)
        self.E_out_2 = T.add(T.pow(self.E_out_r, 2), T.pow(self.E_out_i, 2))

        
    def calculate(self):
        # Take the current value of self.phi and generate the reconstruction:
        if self.intensity_calc is None:
            self.intensity_calc = theano.function(['S_i', 'S_r'], self.E_out_2)
        
        return self.intensity_calc([self.profile_s_r, self.profile_s_i])


def get_centre_range(n):
    # returns the indices to use given an nxn SLM
    # e.g. if 8 pixels, then padding to 16 means the centre starts at 4 -> 12  (0 1 2 3   4 5 6 7 8 9 10 11   12 13 14 15)
    return n/2, n + n/2


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
    
fft = FourierOp()