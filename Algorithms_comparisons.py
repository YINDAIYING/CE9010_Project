import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import os, glob, time, copy, random, zipfile
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook as tqdm
import time
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms
from torch.utils.data import TensorDataset, DataLoader,Dataset

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#set seed to reproduce
seed = 1
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed**2)
    torch.manual_seed(seed**3)
    torch.cuda.manual_seed(seed**4)
set_random_seed(seed)

#load preprocessed train/val data
with open('train_data_3000_shape_64.json', 'r') as fp:
    train_dic = json.load(fp)
train_data, train_target = np.asarray(train_dic['data']), np.asarray(train_dic['target'])
print('Train Data Loaded with shape', train_data.shape)
original_train_shape = train_data.shape
train_data = train_data.reshape(train_data.shape[0], -1)
print('After Reshape:', train_data.shape)

with open('val_data_500_shape_64.json', 'r') as fp:
    val_dic = json.load(fp)
val_data, val_target = np.asarray(val_dic['data']), np.asarray(val_dic['target'])
print('Val Data Loaded with shape', val_data.shape)
original_val_shape = val_data.shape
val_data = val_data.reshape(val_data.shape[0], -1)
print('After Reshape:', val_data.shape)

#SVM
start = time.time()
svm = SVC(gamma='auto')
obj = svm.fit(train_data, train_target)
print('SVM Time Elapsed =', time.time()-start)
print('SVM Train Acc =', obj.score(train_data, train_target))
print('SVM Val Acc =', obj.score(val_data, val_target))

#Logistic Regression
start = time.time()
lr = LogisticRegression(random_state=seed)
obj = lr.fit(train_data, train_target)
print('LR Time Elapsed =', time.time()-start)
print('LR Train Acc =', obj.score(train_data, train_target))
print('LR Val Acc =', obj.score(val_data, val_target))

#Random Forest
start = time.time()
rf = RandomForestClassifier(random_state=seed)
obj = rf.fit(train_data, train_target)
print('RF Time Elapsed =', time.time()-start)
print('RF Train Acc =', obj.score(train_data, train_target))
print('RF Val Acc =', obj.score(val_data, val_target))

#KNN
start = time.time()
knn = KNeighborsClassifier()
obj = knn.fit(train_data, train_target)
print('KNN Time Elapsed =', time.time()-start)
print('KNN Train Acc =', obj.score(train_data, train_target))
print('KNN Val Acc =', obj.score(val_data, val_target))

#Ridge
start = time.time()
ridge = Ridge()
obj = ridge.fit(train_data, train_target)
print('Ridge Time Elapsed =', time.time()-start)

prediction = np.array(obj.predict(train_data)>0.5, dtype = int)
train_acc = accuracy_score(train_target, np.array(prediction).reshape(-1,1))
print('Ridge Train Acc =', train_acc)

prediction = np.array(obj.predict(val_data)>0.5, dtype = int)
val_acc = accuracy_score(val_target, np.array(prediction).reshape(-1,1))
print('Ridge Val Acc =', val_acc)

#Lasso
start = time.time()
lasso = Lasso()
obj = lasso.fit(train_data, train_target)
print('Lasso Time Elapsed =', time.time()-start)

prediction = np.array(obj.predict(train_data)>0.5, dtype = int)
train_acc = accuracy_score(train_target, np.array(prediction).reshape(-1,1))
print('Lasso Train Acc =', train_acc)

prediction = np.array(obj.predict(val_data)>0.5, dtype = int)
val_acc = accuracy_score(val_target, np.array(prediction).reshape(-1,1))
print('Lasso Val Acc =', obj.score(val_data, val_target))

#Neural Network
#customize dataset
class TransformDataset(Dataset):
    def __init__(self,images,labels):
        self.data=torch.Tensor(images)
        self.targets=torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample=self.data[idx]
        target=self.targets[idx]
        return sample,target

#define train function
def train_model(net, dataloader_dict, criterion, optimizer, num_epoch):
    
    since = time.time()
    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch + 1, num_epoch))
        print('-'*20)
        
        for phase in ['train', 'val']:

            if phase == 'train':
                net.train()
            else:
                net.eval()
                
            epoch_loss = 0.0
            epoch_corrects = 0
            
            for inputs, labels in tqdm(dataloader_dict[phase]):
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)
                    
            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

            if phase=='train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(net.state_dict())
                
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    net.load_state_dict(best_model_wts)
    return net

#draw loss/accuracy
def draw_curve(name, train_loss, val_loss, train_acc, val_acc):
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="acc")
    x_epoch = list(range(len(train_acc)))
    ax0.plot(x_epoch, train_loss, 'bo-', label='train')
    ax0.plot(x_epoch, val_loss, 'ro-', label='val')
    ax1.plot(x_epoch, train_acc, 'bo-', label='train')
    ax1.plot(x_epoch, val_acc, 'ro-', label='val')
    ax0.legend()
    ax1.legend()
    fig.savefig(name+'.png')
    plt.close('all')

#NN setting
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = train_data.reshape(original_train_shape)
val_data = val_data.reshape(original_val_shape)

train_dataset = TransformDataset(train_data, train_target)
val_dataset = TransformDataset(val_data, val_target)

train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}

#VGG16
model_kind = 'vgg16'
num_epoch = 5
size = 64

net = torch.load('pretrained_models/pretrained_vgg16.pkl')
net.classifier[0] = nn.Linear(in_features=2048, out_features=512)
net.classifier[3] = nn.Linear(in_features=512, out_features=512)
net.classifier[6] = nn.Linear(in_features=512, out_features=2)
update_params_name = ['classifier.6.weight', 'classifier.6.bias']

net.to(device)

params_to_update = []
for name, param in net.named_parameters():
    if name in update_params_name:
        param.requires_grad = True
        params_to_update.append(param)
        print(name)
    else:
        param.requires_grad = False

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)

train_loss,val_loss,train_acc,val_acc = [],[],[],[]
net = train_model(net, dataloader_dict, criterion, optimizer, num_epoch)
draw_curve(model_kind, train_loss, val_loss, train_acc, val_acc)
