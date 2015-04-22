import ...

# Take code from March expts.ipynb



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
        self.phi    = theano.shared(value=init_phi.astype(theano.config.floatX),
                                    name='phi')
        self.phi_rate = theano.shared(value=np.zeros_like(init_phi).astype(theano.config.floatX),
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
        self.E_out_r, self.E_out_i = FourierOp()(self.E_in_r_pad, self.E_in_i_pad)        
        
        # finally, the output intensity:
        #self.E_out_2 = T.pow(self.E_out_r, 2) + T.pow(self.E_out_i, 2)
        self.E_out_2 = T.add(T.pow(self.E_out_r, 2), T.pow(self.E_out_i, 2))

        
    def calculate(self):
        # Take the current value of self.phi and generate the reconstruction:
        if self.intensity_calc is None:
            self.intensity_calc = theano.function(['S_i', 'S_r'], self.E_out_2)
        
        return self.intensity_calc([self.profile_s_r, self.profile_s_i])