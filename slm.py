import numpy as np
import theano
import theano.tensor as T
from theano.gradient import DisconnectedType

try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    
    def wrap_fft(*args, **kwargs):
        fft2 = pyfftw.interfaces.numpy_fft.fft2(threads=8, *args, **kwargs)
        return fft2
    
    def wrap_ifft(*args, **kwargs):
        ifft2 = pyfftw.interfaces.numpy_fft.ifft2(threads=8, *args, **kwargs)
        return ifft2

    
    fft2_call = wrap_fft
    ifft2_call = wrap_ifft
    
    # assert False
    # pyfftw as implemented fails for currently unknown reasons on the last step?.
    
except:
    fft2_call = np.fft.fft2
    ifft2_call = np.fft.ifft2
    print("Warning: using numpy FFT implementation.  Consider using pyFFTW for faster Fourier transforms.")


class SLM(object):
    
    def __init__(self, target, initial_phi, profile_s=None, A0=1.0):
        self.target = target
        self.n_pixels = int(target.shape[0] / 2)   # target should be 512x512, but SLM pattern calculated should be 256x256.
        self.intensity_calc = None
        
        self.cost = None   # placeholder for cost function.
        
        if profile_s is None:
            profile_s = np.ones((self.n_pixels, self.n_pixels))
            
        assert profile_s.shape == (self.n_pixels, self.n_pixels), 'profile_s is wrong shape, should be ({n},{n})'.format(n=self.n_pixels)
        self.profile_s_r = profile_s.real.astype('float64')
        self.profile_s_i = profile_s.imag.astype('float64')
        
        assert initial_phi.shape == (self.n_pixels**2,), "initial_phi must be a vector of phases of size N^2 (not (N,N)).  Shape is " + str(initial_phi.shape)
        
        self.A0 = A0
        
        # Set zeros matrix:
        self.zero_frame = np.zeros((2*self.n_pixels, 2*self.n_pixels), dtype='float64')
        
        # Phi and its momentum for use in gradient descent with momentum:
        self.phi    = theano.shared(value=initial_phi.astype('float64'),
                                    name='phi')
        self.phi_rate = theano.shared(value=np.zeros_like(initial_phi).astype('float64'),
                                      name='phi_rate')
        
        self.S_r = theano.shared(value=self.profile_s_r,
                                 name='s_r')
        self.S_i = theano.shared(value=self.profile_s_i,
                                 name='s_i')
        self.zero_matrix = theano.shared(value=self.zero_frame,
                                         name='zero_matrix')
        
        # E_in: (n_pixels**2)
        phi_reshaped = self.phi.reshape((self.n_pixels, self.n_pixels))
        self.E_in_r = self.A0 * (self.S_r*T.cos(phi_reshaped) - self.S_i*T.sin(phi_reshaped))
        self.E_in_i = self.A0 * (self.S_i*T.cos(phi_reshaped) + self.S_r*T.sin(phi_reshaped))
        
        # E_in padded: (4n_pixels**2)
        idx_0, idx_1 = get_centre_range(self.n_pixels)
        self.E_in_r_pad = T.set_subtensor(self.zero_matrix[idx_0:idx_1,idx_0:idx_1], self.E_in_r)
        self.E_in_i_pad = T.set_subtensor(self.zero_matrix[idx_0:idx_1,idx_0:idx_1], self.E_in_i)
        
        # E_out:
        self.E_out_r, self.E_out_i = fft(self.E_in_r_pad, self.E_in_i_pad)        
        
        # finally, the output intensity:
        self.E_out_2 = T.add(T.pow(self.E_out_r, 2), T.pow(self.E_out_i, 2))
        
    def get_E_out(self):
        
        f_E_out = theano.function([], [self.E_out_r, self.E_out_i], on_unused_input='warn')
        return f_E_out
        
    def get_intensity(self):
        f_E_out_2 = theano.function([], self.E_out_2, on_unused_input='warn')
        return f_E_out_2

def get_centre_range(n):
    # returns the indices to use given an nxn SLM
    # e.g. if 8 pixels, then padding to 16 means the centre starts at 4 -> 12  (0 1 2 3   4 5 6 7 8 9 10 11   12 13 14 15)
    return int(n/2), int(n + n/2)


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
        #s = np.fft.ifft2(x) * (nx*ny)
        #s = pyfftw.interfaces.numpy_fft.ifft2(x, threads=8) * (nx*ny)
        s = ifft2_call(x) * (nx*ny)
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
        #s = np.fft.fft2(x)  # has "1" normalisation
        #s = pyfftw.interfaces.numpy_fft.fft2(x, threads=8)
        s = fft2_call(x)
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
            print('z_r using zeros_like')
            z_r = z_i.zeros_like()
        
        if isinstance(z_i.type, DisconnectedType):
            print('z_i using zeros_like')
            z_i = z_r.zeros_like()
        
        y = InverseFourierOp()(z_r, z_i)
        return y
    
fft = FourierOp()


def make_target_power2(n, r0, d, A=1.0):
    """
    Create n x n target: 
    2nd order power law centred on r0=(x0,y0) with diameter d and amplitude A
    """
    x = np.array(range(n))*1.
    X, Y = np.meshgrid(x, x)
    
    delta_r2 = np.power(X - r0[0], 2) + np.power(Y - r0[1], 2)
    z = A - 4*A/d**2 * delta_r2
    z[delta_r2 > d**2/4] = 0
    
    return z

def make_gaussian(n, r0, sigma, A=1.0):
    x = np.array(range(n))
    X, Y = np.meshgrid(x, x)
    
    delta_r2 = np.power(X - r0[0], 2) + np.power(Y - r0[1], 2)
    z = A*np.exp(-delta_r2/(2*sigma**2))
    
    return z

def make_weighting(n, r0, d, b, s):
    """
    Create n x n weighting array: 
    Pattern centre r0, diameter d defined by target;
    Border b gives number of pixels between the edge of the trapping pattern and start of noise region
    As a simple example, just have a step change between flat signal region with amplitude 1
    and noise region with amplitude s relative to this.
    """
    x = np.array(range(n))*1.
    X, Y = np.meshgrid(x, x)
    
    delta_r2 = np.power(X - r0[0], 2) + np.power(Y - r0[1], 2)
    z = np.ones((n, n))
    z[delta_r2 > (b + d/2)**2] = s
    
    return z