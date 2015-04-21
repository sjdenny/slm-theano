"""
First attempt at the full works: specify a cost function, compute derivatives,
and run gradient descent to approach a target function.
"""

import time
import numpy as np
import theano
import theano.tensor as T

class slmOptimisation(object):
  """
  Abstract representation of the optimisation process.
  """
  
  def __init__(self, target, init_phi, s_profile=None):
    self.target = target
    self.N = target.shape[0] / 2
    N = self.N
    assert target.shape == (2*N,2*N)
    
    if s_profile is None:
      s_profile = np.ones((N,N))
    self.s_profile = s_profile.astype(theano.config.floatX)
      
    self.cost_function = None
    
    # Set up cost function:
    self.phi    = theano.shared(value=init_phi.astype(theano.config.floatX),
                                name='phi')
    self.phi_rate = theano.shared(value=np.zeros_like(init_phi).astype(theano.config.floatX),
                                name='phi_rate')
    self.S      = T.matrix('S')
    self.zeros  = T.matrix('zeros')
    # TODO need to pad out phi and S
    
    E_in_r = self.pad_params(self.S*T.cos(self.phi), N)
    E_in_i = self.pad_params(self.S*T.sin(self.phi), N)

    arraySize = (N,N)

    cos_tensor, sin_tensor = make_fourier_matrix_fast(2*N)

    E_out_r = (T.tensordot(E_in_r, cos_tensor, axes=[[0, 1], [0, 1]]) +
               T.tensordot(E_in_i, sin_tensor, axes=[[0, 1], [0, 1]])) / N
    E_out_i = (-T.tensordot(E_in_r, sin_tensor, axes=[[0, 1], [0, 1]]) +
               T.tensordot(E_in_i, cos_tensor, axes=[[0, 1], [0, 1]])) / N
  
    self.E_out_2 = T.pow(E_out_r, 2) + T.pow(E_out_i, 2)

  def pad_params(self, E_in, N):
    #centre = T.horizontal_stack(self.zeros, E_in, self.zeros)
    #tb = T.horizontal_stack(self.zeros, self.zeros, self.zeros, self.zeros)
    #padded = T.vertical_stack(tb, centre, tb)

    tb = T.horizontal_stack(E_in, E_in)
    padded = T.vertical_stack(tb, tb)
      
    return padded


def make_target_ring(N, w1=0.2/3.0, w2=0.1/3.0, r=0.5/3.0):
  x = np.linspace(-1.0,1.0,N)
  X, Y = np.meshgrid(x, x)
  
  target = np.exp(-np.power(X,2)/(2*w1**2)) + \
           np.exp(-np.power((np.power(np.power(X,2) + np.power(Y,2), 0.5) - r),2)/(2*w2**2))
  
  return target

def pad_target(target):
  N = target.shape[0]
  centre = np.hstack((np.zeros(shape=(N,N/2)),
                      target,
                      np.zeros(shape=(N,N/2)) ))
  padded_target = np.vstack((np.zeros(shape=(N/2,2*N)),
                             centre,
                             np.zeros(shape=(N/2,2*N)) ))

  return padded_target

def plot_target(target):
  import matplotlib.pyplot as plt
  #plt.close('all')
  plt.figure()
  plt.imshow(target, extent=[-1,1,-1,1])
  plt.colorbar()
  plt.show()
  
def initial_phase(N):
  """ Return a randomised phase array over [0, 2pi]
  """
  return np.random.uniform(low=0, high=2*np.pi, size=(N,N))


def make_fourier_matrix(N):
  cos_tensor = np.zeros(shape=(N,N,N,N), dtype=theano.config.floatX)
  sin_tensor = np.zeros(shape=(N,N,N,N), dtype=theano.config.floatX)
  print 'Starting to construct Fourier matrix...'
  t_start = time.time()
  for p in np.arange(0,N):
    for q in np.arange(0,N):
      for n in np.arange(0,N):
        for m in np.arange(0,N):
          cos_tensor[p,q,n,m] = np.cos(2*np.pi*(p*n+q*m)/N)
          sin_tensor[p,q,n,m] = np.sin(2*np.pi*(p*n+q*m)/N)
  t_end = time.time()
  print '... took {} seconds.'.format(t_end - t_start)
  
  return cos_tensor, sin_tensor

def make_fourier_matrix_fast(N):
  print 'Starting to construct Fourier matrix...'
  cos_tensor = np.zeros(shape=(N,N,N,N), dtype=theano.config.floatX)
  sin_tensor = np.zeros(shape=(N,N,N,N), dtype=theano.config.floatX)
  t_start = time.time()
  n = np.arange(0,N)
  Mi, Ni = np.meshgrid(n,n)
  for p in np.arange(0,N):
    for q in np.arange(0,N):
      cos_tensor[p,q,:,:] = np.cos(2*np.pi*(p*Ni+q*Mi)/N)
      sin_tensor[p,q,:,:] = np.sin(2*np.pi*(p*Ni+q*Mi)/N)
  t_end = time.time()
  print '... took {} seconds.'.format(t_end - t_start)
  
  return cos_tensor, sin_tensor

def main():
  # first, create a target function (Gaussian ring)
  # number of pixels in x and y:
  N = 8
  l_rate = 0.01  # 'learning rate'
  momentum = 0.9 # momentum decay
  target = pad_target(make_target_ring(N, r=0.5, w2=0.2/3));
  np.savetxt('slmOpt1_target.txt', target)
  init_phi = initial_phase(N)
  
  # initialise
  slmOpt = slmOptimisation(target, init_phi)
  # set cost function
  cost_function = T.sum(T.pow(slmOpt.target - slmOpt.E_out_2, 2))
  
  # check we can evaluate it at the start:
  #f_cost = theano.function([slmOpt.S], cost_function)
  
  #C = f_cost(slmOpt.s_profile)
  #print "Initial cost: " + str(C)
  
  # compile a gradient descent function
  grad = T.grad(cost_function, wrt=slmOpt.phi)
  updates = ((slmOpt.phi, slmOpt.phi - l_rate * slmOpt.phi_rate),
             (slmOpt.phi_rate, momentum*slmOpt.phi_rate + (1.-momentum)*grad))
  print "Compiling update function..."
  update = theano.function([slmOpt.S, slmOpt.zeros], 
                           cost_function, 
                           updates=updates,
                           on_unused_input='warn')
  print "...done"
  
  zero_mat = np.zeros(shape=(N/2,N/2), dtype=theano.config.floatX)
  C = update(slmOpt.s_profile, zero_mat)
  print "Initial cost: " + str(C)
  
  last_C = C
  n = 0
  while True:
    n += 1
    cost = update(slmOpt.s_profile, zero_mat)
    if np.mod(n, 100) == 0:
      if not last_C > 1.00001 * cost:
        # if cost hasn't decreased by more than 1% in 100 iterations..
        break
      else:
        last_C = cost
        print n, cost
  
  # create reconstruction of target:
  print "Compiling reconstruction function..."
  slmOut = theano.function([slmOpt.S], slmOpt.E_out_2)
  print "...done"
  slmOut(slmOpt.s_profile)
  
  # return parameters:
  return slmOpt.phi.get_value(), slmOut(slmOpt.s_profile)
  

if __name__ == '__main__':
  params, output = main()
  # save output:
  np.savetxt('slmOpt1_output.txt', output)
