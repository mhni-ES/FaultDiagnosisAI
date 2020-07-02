
import numpy as np
import scipy.io as sio
import os

from sklearn import decomposition


def load_file(input):
    item = sio.loadmat(input)
    IQ = item['IQ']
    return IQ.reshape((num_samples, int(5000000 / num_samples)))


num_samples = 1000
filename_load = '/Users/mhni/Desktop/GOMX5-MARK3/NeuralNetwork/AMO/test_data.mat'

IQ = np.zeros((num_samples, 2, 5000))
IQ_raw = load_file(filename_load)
IQ[ :, 0, :] = np.real(IQ_raw)
IQ[ :, 1, :] = np.imag(IQ_raw)
        ##
if not os.path.exists('test_data/' ):
    os.makedirs('test_data/' )

num = 0
for k in range(IQ.shape[2]):
    IQ_save = IQ[:, k, :]
    filename_save = 'test_data/'  + str(num) + '.npy'
    np.save(filename_save, IQ_save)
    num = num +1