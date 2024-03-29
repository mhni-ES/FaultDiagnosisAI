import torch
import json

from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import numpy as np
import time

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn import decomposition

import scipy.io as sio

np.random.seed(0)
loss_vec = []
# Seed for torch cuda optimization
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyper-parameters
num_classes = 33 # this is the total number of classes that can be diagnosed right now this will grow exponetially the more requirements i do.
num_epochs = 50
batch_size = 200
learning_rate = 0.1


def npy_load_file(input):
    item = np.load(input)
    return item


def mat_load_file(input):
    raw = sio.loadmat(input)
    IQ_raw = raw['IQ']
    IQ_raw = IQ_raw.reshape(1000, -1)
    IQ_raw = (np.abs(IQ_raw).mean(0).reshape(-1))
    #    length = (IQ_raw).shape[1]
    #    item = np.zeros(length*2)
    #    item[0:length] = np.real(IQ_raw)
    #    item[length:] = np.imag(IQ_raw)
    # return item[::1000]
    return IQ_raw

def noise_train(input):
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
    #print(noise_avg_db)
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    #noise_avg_watts = noise_avg_watts * noise_avg_watts

    # Generate an sample of white noise
    mean_noise = 0
    variance = np.sqrt(noise_avg_watts)#(0.001 * 0.5)
    #print(variance)
    noise = mean_noise + variance * torch.randn(10000)
    SNR =10* np.log10(sig_mean / torch.mean(torch.abs(noise)))
   # print(SNR.numpy())
    # Noise up the original signal
    # (0.1 ** 0.5) * torch.randn(10000)
    output = input + noise
    return output

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
    #print(noise_avg_db)
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    #noise_avg_watts = noise_avg_watts * noise_avg_watts

    # Generate an sample of white noise
    mean_noise = 0
    variance = np.sqrt(noise_avg_watts)#(0.001 * 0.5)
    #print(variance)
    noise = mean_noise + variance * torch.randn(len(input))
    SNR =10* np.log10(sig_mean / torch.mean(torch.abs(noise)))
    #print(SNR.numpy())
    # Noise up the original signal
    # (0.1 ** 0.5) * torch.randn(10000)
    output = input + noise
    return output
# a = mat_load_file(filename_load)
train_dataset = datasets.DatasetFolder(root='data_processed_big/1000/training', loader=npy_load_file, extensions='npy')
test_dataset = datasets.DatasetFolder(root='data_processed_big/1000/testing', loader=npy_load_file, extensions='npy')
classes = list(train_dataset.class_to_idx.keys())
dataset_len = len(train_dataset)
train_dataset, test_dataset = torch.utils.data.random_split(train_dataset,
                                                            (int(dataset_len * 0.7), int(dataset_len * 0.3)))
# train_dataset = datasets.DatasetFolder(root='data',loader=mat_load_file, extensions='mat')
# test_dataset = datasets.DatasetFolder(root='data',loader=mat_load_file, extensions='mat')


# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
batch_size_test = 10000
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size_test,
                                          shuffle=True)

# Apply data split: torch.utils.data.random_split(dataset, lengths)

# Logistic regression model
input_size = 2 * 5000
# input_size = 100

import torch.nn as nn
import torch.nn.functional as F


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Sequential(nn.BatchNorm1d(input_size), nn.Linear(input_size, hidden_size),nn.BatchNorm1d(hidden_size), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.BatchNorm1d(num_classes))
        # self.bn1 = nn.BatchNorm1d(hidden_size)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.bn2 = nn.BatchNorm1d(hidden_size)
        # self.fc3 = nn.Linear(hidden_size, hidden_size)
        # self.bn3 = nn.BatchNorm1d(hidden_size)
        # self.fc4 = nn.Linear(hidden_size, hidden_size)
        # self.bn4 = nn.BatchNorm1d(hidden_size)
        # self.fc5 = nn.Linear(hidden_size, num_classes)
        # self.dropout = nn.Dropout(p = 0.00)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


model = NeuralNet(input_size=input_size, hidden_size=500, num_classes=num_classes)
model.to(dev)  # moves model to the CUDA core if available
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# Scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = num_epochs / 5, gamma = 0.1, last_epoch = -1)
Scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)
# Train the model
Start = time.time()
total_step = len(train_loader)

for epoch in range(num_epochs):
    correct = 0
    total = 0
    for i, (data, labels) in enumerate(train_loader):
        # print(i)
        # Reshape images to (batch_size, input_size)
        data = torch.reshape(data, (-1, input_size))

        # Forward pass
        input = noise_train(data.float())
        outputs = model(input)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_vec.append(loss.item())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    accuracy = 100 * correct / total

    #    if (i+1) % 1 == 0:
    #        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
    #               .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    print('Epoch [{}/{}],Loss: {:.4f},Accuracy: {:.2f}'
          .format(epoch + 1, num_epochs, loss.item(), accuracy))
    Scheduler.step(accuracy)

print('Time use:', time.time() - Start)
# for i, (data_test, labels) in enumerate(test_loader):
(data_test, labels_test) = next(iter(test_loader))
# Reshape images to (batch_size, input_size)
data_test = torch.reshape(data_test, (-1, input_size))

# Forward pass
input = noise(data_test.float())
outputs = model(input)
loss = criterion(outputs, labels_test)
# Find accuracy of the test
_, predicted = torch.max(outputs.data, 1)
test_total = labels_test.size(0)
test_correct = (predicted == labels_test).sum()
accuracy_test = 100 * test_correct.numpy() / test_total

print('loss is ', loss.item())
print('accuracy is ', accuracy_test)

torch.save(model.state_dict(), '/Users/mhni/Desktop/GOMX5-MARK3/NeuralNetwork/AMO/AMO_NN_big_50epoch_SNR11.pkl')


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
    np.savetxt('confusionMatbig50epoch_SNR11.txt', cm, delimiter=',')
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
classes = ['4', '3', '2', '5', '12', '15', '14', '13', '9', 'No Faults', '7', '6', '1', '8', '16', '11', '10','4 ATT', '3ATT', '2ATT', '5ATT', '12ATT', '15ATT', '14ATT', '13ATT', '9ATT', 'No Faults', '7ATT', '6ATT', '1ATT', '8ATT', '16ATT', '11ATT', '10ATT']
plot_confusion_matrix(labels_test, outputs.argmax(1).detach().numpy(), classes=classes_int, normalize='true',
                      labels=classes)
plt.savefig('conf_mat_big_50epoch_SNR11.pdf')


plt.figure()
plt.plot(loss_vec)
plt.xlabel('Number of training steps')
plt.ylabel('Cross entropy loss')
plt.savefig('Loss_vec_big_50epoch_SNR11.pdf')

plt.figure()
for i in range(7):
    plt.scatter((data[labels == i])[:, 0], (data[labels == i])[:, 1])
    plt.scatter((data_test[labels_test == i])[:, 0], (data_test[labels_test == i])[:, 1])
plt.show()
# Save the model checkpoint
#torch.save(model.state_dict(), 'model.ckpt')