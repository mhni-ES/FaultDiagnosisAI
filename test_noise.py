import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import Dataset


import numpy as np
import time

from matplotlib import pyplot as plt


from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


import scipy.io as sio

#np.random.seed(0)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

def npy_load_file(input):
	item = np.load(input)
	return item

def mat_load_file(input):
    raw = sio.loadmat(input)
    IQ = np.zeros((1000, 2, 5000))
    IQ_raw = raw['IQ']
    IQ_raw = IQ_raw.reshape((1000, int(5000000 / 1000)))

    IQ[:, 0, :] = np.real(IQ_raw)
    IQ[:, 1, :] = np.imag(IQ_raw)

    IQ_save = IQ
    #IQ_save = IQ_save.reshape(1000, -1)
    #IQ_save = (np.abs(IQ_save).mean(0).reshape(-1))
    #    length = (IQ_raw).shape[1]
    #    item = np.zeros(length*2)
    #    item[0:length] = np.real(IQ_raw)
    #    item[length:] = np.imag(IQ_raw)
    # return item[::1000]
    return IQ_save


import math as mt


def mostFrequent(arr, n):
    # Insert all elements in Hash.
    Hash = dict()
    for i in range(n):
        if arr[i] in Hash.keys():
            Hash[arr[i]] += 1
        else:
            Hash[arr[i]] = 1

    # find the max frequency
    max_count = 0
    res = -1
    for i in Hash:
        if (max_count < Hash[i]):
            res = i
            max_count = Hash[i]

    return res, max_count

PATH = '/Users/mhni/Desktop/GOMX5-MARK3/NeuralNetwork/AMO/AMO_NN_big_50epoch_SNR5.pkl'
batch_size = 100
batch_size_test = 10000
input_size = 2 * 5000
num_classes = 33 # this is the total number of classes that can be diagnosed right now this will grow exponetially the more requirements i do.
num_epochs = 50
batch_size = 200
learning_rate = 0.1
hidden_size = 500
train_dataset = datasets.DatasetFolder(root='data_processed_big/1000/training', loader=npy_load_file, extensions='npy')
#train_dataset = datasets.DatasetFolder(root='data_processed_big_noise/1000/training', loader=npy_load_file, extensions='npy')
classes = list(train_dataset.class_to_idx.keys())
dataset_len = len(train_dataset)
train_dataset, test_dataset = torch.utils.data.random_split(train_dataset,
                                                            (int(dataset_len * 0.7), int(dataset_len * 0.3)))
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size_test,
                                          shuffle=False)
#Python fail at test dB 6 but not a single class has been predicted correct compared to the origianl class.

# add noise based on SNR to the test_data set
# Adding noise using target SNR

def noise(input):
    # need to figure out how to make SNR here... this is how i should add it finally!
    target_snr_db = 11
    # Calculate signal power and convert to dB
    sig_mean = torch.mean(torch.abs(input))
    #sig_mean = 0.0282 # signal mean from matlab results.... need to look into the mean of the signal
    #print(sig_mean)
    #print(sig_mean.numpy())
    sig_avg_db = 10 * np.log10(abs(sig_mean))
    #sig_avg_db = 0
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    print(noise_avg_db)
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    #noise_avg_watts = noise_avg_watts * noise_avg_watts

    # Generate an sample of white noise
    mean_noise = 0
    variance = np.sqrt(noise_avg_watts)#(0.001 * 0.5)
    print(variance)
    noise = mean_noise + variance * torch.randn(len(input))
    SNR =10* np.log10(sig_mean / torch.mean(torch.abs(noise)))
    print(SNR.numpy())
    # Noise up the original signal
    # (0.1 ** 0.5) * torch.randn(10000)
    output = input + noise
    return output

## AI neural net
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Sequential(nn.BatchNorm1d(input_size), nn.Linear(input_size, hidden_size),nn.BatchNorm1d(hidden_size), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.BatchNorm1d(num_classes))


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
#test_dataset = datasets.DatasetFolder(root = 'data', loader=npy_load_file, extensions = 'npy')

#test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                          batch_size=batch_size_test,
#                                          shuffle=True)

#classes = list(test_dataset.class_to_idx.keys())
model = NeuralNet(input_size=input_size, hidden_size=500, num_classes=num_classes)
model.load_state_dict(torch.load(PATH))
model.to(dev)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
model.eval()


(data_test, labels_test) = next(iter(test_loader))
# Reshape images to (batch_size, input_size)
data_test = torch.reshape(data_test, (-1, input_size))
input_data = noise(data_test.float()) #data_test.float() #
outputs = model(input_data)
loss = criterion(outputs, labels_test)

def plot_confusion_matrix(y_true, y_pred, classes, labels=None,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    #np.savetxt('confusion_trainedNoise11SNR_8.txt', cm, delimiter=',')
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.ylim([1.5, -.5])
    fig.tight_layout()
    return ax


classes_int = np.arange(len(classes))
classes_label = ['4', '3', '2', '5', '12', '15', '14', '13', '9', 'No Faults', '7', '6', '1', '8', '16', '11', '10','4 ATT', '3ATT', '2ATT', '5ATT', '12ATT', '15ATT', '14ATT', '13ATT', '9ATT', 'No Faults', '7ATT', '6ATT', '1ATT', '8ATT', '16ATT', '11ATT', '10ATT']
plot_confusion_matrix(labels_test, outputs.argmax(1).detach().numpy(), classes=classes_int, normalize='true',
                      labels=classes_label)