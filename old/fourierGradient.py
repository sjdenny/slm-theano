import numpy as np
from scipy.fftpack import fft2, fftshift
import matplotlib.pyplot as plt


# make a sample input:
arraySize = (8,8)
X = 2*np.pi*np.random.uniform(size=arraySize)

eIn = np.exp(1j*X)
eOut = fftshift(fft2(eIn))