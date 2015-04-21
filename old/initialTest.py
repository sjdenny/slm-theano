import time
import numpy as np
from scipy.fftpack import fft2, fftshift
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

# make a sample input:
arraySize = (8,8)
X = 2*np.pi*np.random.uniform(size=arraySize)

eIn = np.exp(1j*X)
eOut = fftshift(fft2(eIn))

# visualise output from random input:
#plt.close('all')
#fig = plt.figure()
#plt.pcolor(abs(eOut)**2)
#plt.show()


# Now make a theano fourier transform:
x = T.vector('x') # input phase array, as a vector.
M = T.tensor3('M')
eOut = T.matrix('eOut')

N = arraySize[0]
fft_matrix = np.zeros((N,N), dtype=complex)
# TODO this is a terrible way to construct...
for k in np.arange(0,N):
  for m in np.arange(0,N):
    fft_matrix[k,m] = np.exp(2*np.pi*1j*k*m/N)

# combine a pair:
fft_tensor = np.zeros((N,N,N), dtype=complex)
for n in np.arange(0,N):
  for m in np.arange(0,N):
    fft_tensor[k,n,m] = fft_matrix[k,n]*np.conj(fft_matrix[k,m])
    
# permute last two indices and sum: will remove complex parts:
fft_tensor = 0.5*(fft_tensor + fft_tensor.swapaxes(1,2)).astype(float)
#fft_tensor.dtype = theano.config.floatX
#eOut = T.tensordot(M, x, axes=[[1], [0]])
eOut = T.tensordot(x,x, axes=[[0],[0]])
X = 2*np.pi*np.random.uniform(size=N)
#X.dtype = theano.config.floatX
f = theano.function([M, x], eOut, on_unused_input='warn')

print "Compare inner products:"
print f(fft_tensor.astype(theano.config.floatX), X.astype(theano.config.floatX))
print np.dot(X,X)


# incorporate matrix:
eOut2 = T.tensordot(M, x, axes=[[1], [0]])
eOut3 = T.tensordot(x, eOut2, axes=[[0], [1]])
f2 = theano.function([M, x], eOut3, on_unused_input='warn')
print f2(fft_tensor.astype(theano.config.floatX), X.astype(theano.config.floatX))

#################
#  new attempt  #
#################

phi    = T.matrix('phi')
S      = T.matrix('S')
E_in_r = S*T.cos(phi)
E_in_i = S*T.sin(phi)

N = 32
arraySize = (N,N)


f_E_in_r = theano.function([phi, S], E_in_r)
f_E_in_r(np.array([[0, 0, 0]], dtype='float32'),
         np.array([[0, 1, 2]], dtype='float32'))

cos_tensor = np.zeros(shape=(N,N,N,N), dtype=theano.config.floatX)
sin_tensor = np.zeros(shape=(N,N,N,N), dtype=theano.config.floatX)
for p in np.arange(0,N):
  for q in np.arange(0,N):
    for n in np.arange(0,N):
      for m in np.arange(0,N):
        cos_tensor[p,q,n,m] = np.cos(2*np.pi*(p*n+q*m)/N)
        sin_tensor[p,q,n,m] = np.sin(2*np.pi*(p*n+q*m)/N)

E_out_r = (T.tensordot(E_in_r, cos_tensor, axes=[[0, 1], [0, 1]]) +
           T.tensordot(E_in_i, sin_tensor, axes=[[0, 1], [0, 1]])) / N
E_out_i = (-T.tensordot(E_in_r, sin_tensor, axes=[[0, 1], [0, 1]]) +
           T.tensordot(E_in_i, cos_tensor, axes=[[0, 1], [0, 1]])) / N

E_out_2 = T.pow(E_out_r, 2) + T.pow(E_out_i, 2)


testArray = 2*np.pi*np.random.uniform(size=arraySize).astype(theano.config.floatX)
f_E_out_r = theano.function([phi, S], E_out_r)
print f_E_out_r(testArray, testArray)

f_E_out = theano.function([phi, S], E_out_2)
print f_E_out(testArray, testArray)


####################
#  Gradient time!  #
####################

target = np.random.uniform(size=arraySize).astype(theano.config.floatX)
cost = T.sum(T.pow(target - E_out_2, 2))

start = time.time()
f_cost = theano.function([phi, S], cost)
end = time.time()
print f_cost(testArray, testArray)
print 'Compiling cost function took {} seconds.'.format(end - start)

g_cost = T.grad(cost, wrt=phi)
start = time.time()
fg_cost = theano.function([phi, S], g_cost)
end = time.time()
print fg_cost(testArray, testArray)
print 'Compiling cost gradient took {} seconds.'.format(end - start)

"""

eOut = fftshift(fft2(eIn));

CT = cbrewer('seq', 'Greys', 100);

plot:
figure(1); clf;
colormap(CT)
imagesc(abs(eOut).^2)
colorbar

%% Make a target:
x = linspace(-1,1,arraySize);
[X, Y] = meshgrid(x, x);

w1 = 0.2/3;
w2 = 0.1/3;
r = +0.5/3;
target = exp(-X.^2/(2*w1^2)) + ...
         exp(-( sqrt(X.^2+Y.^2)-r ).^2/(2*w2^2));

figure(2); clf;
colormap(CT)
imagesc(x,x,target)
colorbar
title('Target')

"""