import numpy as np
import scipy.io as sio
import os

from sklearn import decomposition

def load_file(input):
    item = sio.loadmat(input)
    IQ = item['IQ']
    return IQ.reshape((num_samples, int(5000000 / num_samples)))


num_samples = 1000
version = ''
folders = os.listdir('data_big_noise')

IQ = np.zeros((33, 10, num_samples, 2, 5000))
for f_num, folder in enumerate(folders):
    for i in range(0, 10):
        if i in range(9):
            version = 'training/'
        else:
            version = 'testing/'
        filename_load = 'data_big_noise/' + folder + '/' + str(i + 1) + '.mat'
        print(filename_load)
        IQ_raw = load_file(filename_load)

        #        IQ_raw = np.abs(np.fft.fftshift(np.fft.fft(IQ_raw,axis=1),1))
        #        IQ[f_num,i,:,0,:] = np.real(IQ_raw)
        #        IQ[f_num,i,:,1,:] = np.real(IQ_raw)
        #
        #
        IQ[f_num, i, :, 0, :] = np.real(IQ_raw)
        IQ[f_num, i, :, 1, :] = np.imag(IQ_raw)
        ##
        if not os.path.exists('data_processed_big_noise/1000/' + version + folder):
            os.makedirs('data_processed_big_noise/1000/' + version + folder)
    num = 0
    for i in range(0, 10):
        if i in range(9):
            version = 'training/'
        else:
            version = 'testing/'
            num = 0

        for k in range(IQ.shape[2]):
            IQ_save = IQ[f_num, i, k, :]
            filename_save = 'data_processed_big_noise/1000/' + version + folder + '/' + str(num) + '.npy'
            np.save(filename_save, IQ_save)
            num = num + 1

# pca_dims = 100
# pca = decomposition.PCA(pca_dims)
# print('Finding PCA')
# pca.fit(IQ.reshape(-1,1000))

# #ica = decomposition.FastICA(n_components=10,max_iter = 25, random_state=12)
# #print('Finding ICA')
# #ica.fit(IQ.reshape(-1,10000))


# for f_num,folder in enumerate(folders):
#     for i in range(0,3):
#         if i == 0:
#             version = 'training_pca' + str(pca_dims)+ '/'
#         if i == 1:
#             version = 'testing_pca' + str(pca_dims) + '/'
#         if i == 2:
#             version = 'validation_pca' + str(pca_dims) + '/'
#         if not os.path.exists('data_processed/' + version + folder):
#             os.makedirs('data_processed/' + version + folder)

#         for k in range(IQ.shape[2]):
#             IQ_save = IQ[f_num,i,k,:].reshape(1,100000)
#             IQ_save = pca.transform(IQ_save)
#             filename_save = 'data_processed/' + version + folder + '/' + str(k) + '.npy'
#             np.save(filename_save,IQ_save)

#