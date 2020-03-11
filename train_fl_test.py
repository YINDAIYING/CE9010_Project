# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import os
from model import ft_net, ft_net_dense, ft_net_NAS, PCB
from random_erasing import RandomErasing
import yaml
import math
from shutil import copyfile
import random
import numpy as np
import scipy.io
version =  torch.__version__
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError:
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--project_dir',default='.', type=str, help='project path')
parser.add_argument('--data_dir',default='data',type=str, help='training dir path')
parser.add_argument('--datasets',default='Market,DukeMTMC-reID,cuhk03-np-detected',type=str, help='e.g. 0 0,1,2')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--use_NAS', action='store_true', help='use NAS' )
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50' )
parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory' )

#arguments for testing federated model
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--test_dir',default='data/fused',type=str, help='./test_data')

opt = parser.parse_args()

opt.local_epoch = 1
opt.no_clients_per_round=3

fp16 = opt.fp16
data_dir = opt.data_dir
project_dir = opt.project_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed**2)
    torch.manual_seed(seed**3)
    torch.cuda.manual_seed(seed**4)
set_random_seed(1)

def get_optimizer(model, lr):
    if not opt.PCB:
        ignored_params = list(map(id, model.classifier.parameters() ))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        optimizer_ft = optim.SGD([
                {'params': base_params, 'lr': 0.1*lr},
                {'params': model.classifier.parameters(), 'lr': lr}
            ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    else:
        ignored_params = list(map(id, model.model.fc.parameters() ))
        ignored_params += (list(map(id, model.classifier0.parameters() ))
                        +list(map(id, model.classifier1.parameters() ))
                        +list(map(id, model.classifier2.parameters() ))
                        +list(map(id, model.classifier3.parameters() ))
                        +list(map(id, model.classifier4.parameters() ))
                        +list(map(id, model.classifier5.parameters() ))
                        #+list(map(id, model.classifier6.parameters() ))
                        #+list(map(id, model.classifier7.parameters() ))
                        )
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        optimizer_ft = optim.SGD([
                {'params': base_params, 'lr': 0.1*lr},
                {'params': model.model.fc.parameters(), 'lr': lr},
                {'params': model.classifier0.parameters(), 'lr': lr},
                {'params': model.classifier1.parameters(), 'lr': lr},
                {'params': model.classifier2.parameters(), 'lr': lr},
                {'params': model.classifier3.parameters(), 'lr': lr},
                {'params': model.classifier4.parameters(), 'lr': lr},
                {'params': model.classifier5.parameters(), 'lr': lr},
                #{'params': model.classifier6.parameters(), 'lr': 0.01},
                #{'params': model.classifier7.parameters(), 'lr': 0.01}
            ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    return optimizer_ft

def save_network(network, cid, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    dir_name = os.path.join(project_dir,'model',name,cid)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    save_path = os.path.join(project_dir,'model',name,cid,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])

def get_model(class_names_size):
    if opt.use_dense:
        model = ft_net_dense(class_names_size, opt.droprate)
    elif opt.use_NAS:
        model = ft_net_NAS(class_names_size, opt.droprate)
    else:
        model = ft_net(class_names_size, opt.droprate, opt.stride)
    if opt.PCB:
        model = PCB(class_names_size)
    return model

def add_model(dst_model, src_model,dst_no_data,src_no_data):
    if (dst_model==None):
        return src_model
    params1 = src_model.named_parameters()
    params2 = dst_model.named_parameters()
    dict_params2 = dict(params2)
    with torch.no_grad():
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].set_(param1.data*src_no_data + dict_params2[name1].data*dst_no_data)
    return dst_model

def scale_model(model, scale):
    params = model.named_parameters()
    dict_params = dict(params)
    with torch.no_grad():
        for name, param in dict_params.items():
            dict_params[name].set_(dict_params[name].data * scale)
    return model


def federated_avg(models,data_num):
    if models==[]:
        return None
    model = add_model(None, models[0],0,data_num[0])
    total_no_data=data_num[0]
    for i in range(1,len(models)):
        model = add_model(model, models[i],total_no_data,data_num[i])
        model = scale_model(model, 1.0 / (total_no_data+data_num[i]))
        total_no_data=total_no_data+data_num[i]
    return model

#functions for testing federated model
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        ff = torch.FloatTensor(n,512).zero_().cuda()
        if opt.PCB:
            ff = torch.FloatTensor(n,2048,6).zero_().cuda() # we have six parts

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img)
                ff += outputs
        # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


datas = [opt.datasets]
if opt.datasets:
    datas = opt.datasets.split(',')

transform_train_list = [
        transforms.Resize((256,128), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_val_list = [
        transforms.Resize(size=(256,128),interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.PCB:
    transform_train_list = [
        transforms.Resize((384,192), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    transform_val_list = [
        transforms.Resize(size=(384,192),interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.erasing_p>0:
    transform_train_list = transform_train_list +  [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
}

train_all = ''
if opt.train_all:
     train_all = '_all'

train_dataloaders = {}
val_dataloaders = {}
train_dataset_sizes = {}
val_dataset_sizes = {}
class_names_size = {}
clients_list = []

for dataset in datas:
    clients_list.append(dataset)
    data_path = os.path.join(data_dir, dataset, 'pytorch')

    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(os.path.join(data_path, 'train' + train_all), data_transforms['train'])
    image_datasets['val'] = datasets.ImageFolder(os.path.join(data_path, 'val'), data_transforms['val'])

    loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                shuffle=True, num_workers=8, pin_memory=True)
                for x in ['train', 'val']}

    train_dataloaders[dataset] = loaders['train']
    val_dataloaders[dataset] = loaders['val']

    train_dataset_sizes[dataset] = len(image_datasets['train'])
    val_dataset_sizes[dataset] = len(image_datasets['val'])
    class_names_size[dataset] = len(image_datasets['train'].classes)

class Client():
    def __init__(self,cid,train_loader, val_loader):
        self.cid=cid
        self.datanum=train_dataset_sizes[cid]
        self.datanum_val = val_dataset_sizes[cid]
        self.dataset_sizes = {'train': self.datanum, 'val':self.datanum_val}
        self.loaders = {'train': train_loader, 'val': val_loader}

        self.full_model=get_model(class_names_size[cid])
        self.classifier = self.full_model.classifier.classifier
        self.full_model.classifier.classifier = nn.Sequential()
        self.model = self.full_model
                
    def train(self,num_epochs,federated_model, global_epoch):
        self.y_err = {'train':[], 'val':[]}
        self.y_loss = {'train':[], 'val':[]}

        self.model.load_state_dict(federated_model.state_dict())
        self.model.classifier.classifier = self.classifier
        self.model = self.model.to(device)
        optimizer=get_optimizer(self.model, opt.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        
        if fp16:
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level = "O1")
        
        criterion = nn.CrossEntropyLoss()

        warm_up = 0.1
        warm_iteration = round(self.datanum/opt.batchsize)*opt.warm_epoch
        since = time.time()

        print('Client', self.cid, 'start training')
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    self.model.train(True)
                else:
                    self.model.train(False)
                running_loss = 0.0
                running_corrects = 0.0
                
                for data in self.loaders[phase]:
                    inputs, labels = data
                    now_batch_size,c,h,w = inputs.shape
                    if now_batch_size<opt.batchsize:
                        continue
                    if use_cuda:
                        inputs = Variable(inputs.cuda().detach())
                        labels = Variable(labels.cuda().detach())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)
                   
                    optimizer.zero_grad()

                    if phase == 'val':
                        with torch.no_grad():
                            outputs = self.model(inputs)
                    else:
                        outputs = self.model(inputs)

                    if not opt.PCB:
                        _, preds = torch.max(outputs.data, 1)
                        loss = criterion(outputs, labels)
                    else:
                        part = {}
                        sm = nn.Softmax(dim=1)
                        num_part = 6
                        for i in range(num_part):
                            part[i] = outputs[i]

                        score = sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) +sm(part[4]) +sm(part[5])
                        _, preds = torch.max(score.data, 1)

                        loss = criterion(part[0], labels)
                        for i in range(num_part-1):
                            loss += criterion(part[i+1], labels)

                    if epoch<opt.warm_epoch and phase == 'train':
                        warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                        loss *= warm_up

                    if phase == 'train':
                        if fp16:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()
                        optimizer.step()

                    if int(version[0])>0 or int(version[2]) > 3: # for the new version like 0.4.0, 0.5.0 and 1.0.0
                        running_loss += loss.item() * now_batch_size
                    else :  # for the old version like 0.3.0 and 0.3.1
                        running_loss += loss.data[0] * now_batch_size
                    running_corrects += float(torch.sum(preds == labels.data))

                epoch_loss = running_loss / (self.dataset_sizes[phase]-self.dataset_sizes[phase]%opt.batchsize)
                epoch_acc = running_corrects / (self.dataset_sizes[phase]-self.dataset_sizes[phase]%opt.batchsize)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                self.y_loss[phase].append(epoch_loss)
                self.y_err[phase].append(1.0-epoch_acc)

                if phase == 'val':
                    last_model_wts = self.model.state_dict()

            time_elapsed = time.time() - since
            print('Client', self.cid, ' Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

        time_elapsed = time.time() - since
        print('Client', self.cid, 'Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

        self.model.load_state_dict(last_model_wts)
        save_network(self.model, self.cid, 'last')

        #self.draw_curve(self.cid, num_epochs, global_epoch, self.y_loss, self.y_err)
        
        self.classifier = self.model.classifier.classifier
        self.model.classifier.classifier = nn.Sequential()
   
    def get_model(self):
        self.model.classifier.classifier = nn.Sequential()
        return self.model
    def get_data_num(self):
        return self.datanum
    def get_train_loss(self):
        return self.y_loss['train'][-1]
    def get_val_loss(self):
        return self.y_loss['val'][-1]
    # def draw_curve(self, cid, num_epochs, current_global_epoch, y_loss, y_err):
    #     fig = plt.figure()
    #     ax0 = fig.add_subplot(121, title="loss")
    #     ax1 = fig.add_subplot(122, title="top1err")
    #     x_epoch = list(range(num_epochs))
    #     ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    #     ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    #     ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    #     ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    #     ax0.legend()
    #     ax1.legend()
    #     dir_name = os.path.join(project_dir,'model',name,cid)
    #     if not os.path.isdir(dir_name):
    #         os.mkdir(dir_name)
    #     fig.savefig(os.path.join(project_dir,'model',name,cid,'Epoch'+str(current_global_epoch)+'_train.jpg'))
    #     plt.close('all')

Clients = {}
for uid in clients_list:
    Clients[uid]=Client(uid, train_dataloaders[uid], val_dataloaders[uid]) 

class Server():
    def __init__(self):
        self.full_model = get_model(750).to(device)
        self.full_model.classifier.classifier = nn.Sequential()
        self.federated_model=self.full_model
        self.federated_model.eval()
        self.train_loss = []
        self.val_loss = []

    def train(self,epoch):
        models=[]
        data_num=[]
        lossTot_train=0
        lossTot_val=0

        current_clients_list = random.sample(clients_list, opt.no_clients_per_round)
        for i in current_clients_list:
            Clients[i].train(opt.local_epoch,self.federated_model, epoch)
        
        for i in current_clients_list:
            lossTot_train+=Clients[i].get_train_loss()
            lossTot_val+=Clients[i].get_val_loss()
            data_num.append(Clients[i].get_data_num())
            models.append(Clients[i].get_model())
        print("==============================")
        print("number of clients used:", len(models))
        print('Train Epoch: {}, AVG Train Loss among clients of lost epoch: {:.6f}'.format(epoch, lossTot_train/opt.no_clients_per_round))
        print('Val Epoch: {}, AVG Val Loss among clients of lost epoch: {:.6f}'.format(epoch, lossTot_val/opt.no_clients_per_round))
        print()
        self.train_loss.append(lossTot_train/opt.no_clients_per_round)
        self.val_loss.append(lossTot_val/opt.no_clients_per_round)
        self.federated_model=federated_avg(models,data_num)
    def draw_curve(self):
        plt.figure()
        x_epoch = list(range(len(self.train_loss)))
        plt.plot(x_epoch, self.train_loss, 'bo-', label='train')
        plt.plot(x_epoch, self.val_loss, 'ro-', label='val')
        plt.legend()
        dir_name = os.path.join(project_dir,'model',name)
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        plt.savefig(os.path.join(project_dir,'model',name,'train.jpg'))
        plt.close('all')
    def test(self):
        test_dir = opt.test_dir
        print("="*10)
        print("Start Tesing!")
        print("="*10)
        print('We use the scale: %s'%opt.ms)
        print()

        test_data_transforms = transforms.Compose([
                transforms.Resize((256,128), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if opt.PCB:
            test_data_transforms = transforms.Compose([
                transforms.Resize((384,192), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        if opt.multi:
            test_image_datasets = {x: datasets.ImageFolder( os.path.join(test_dir,x) ,test_data_transforms) for x in ['gallery','query','multi-query']}
            test_dataloaders = {x: torch.utils.data.DataLoader(test_image_datasets[x], batch_size=opt.batchsize,
                                                    shuffle=False, num_workers=16) for x in ['gallery','query','multi-query']}
        else:
            test_image_datasets = {x: datasets.ImageFolder( os.path.join(test_dir,x) ,test_data_transforms) for x in ['gallery','query']}
            test_dataloaders = {x: torch.utils.data.DataLoader(test_image_datasets[x], batch_size=opt.batchsize,
                                                    shuffle=False, num_workers=16) for x in ['gallery','query']}
        test_class_names = test_image_datasets['query'].classes

        gallery_path = test_image_datasets['gallery'].imgs
        query_path = test_image_datasets['query'].imgs

        gallery_cam,gallery_label = get_id(gallery_path)
        query_cam,query_label = get_id(query_path)

        if opt.multi:
            mquery_path = test_image_datasets['multi-query'].imgs
            mquery_cam,mquery_label = get_id(mquery_path)
                
        self.federated_model = self.federated_model.eval()
        if use_cuda:
            self.federated_model = self.federated_model.cuda()
        
        with torch.no_grad():
            gallery_feature = extract_feature(self.federated_model,test_dataloaders['gallery'])
            query_feature = extract_feature(self.federated_model,test_dataloaders['query'])
            if opt.multi:
                mquery_feature = extract_feature(self.federated_model,test_dataloaders['multi-query'])
        
        result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
        scipy.io.savemat('pytorch_result.mat',result)
        print(opt.name)
        result = project_dir + '/model/%s/result.txt'%opt.name
        os.system('python %s/evaluate_gpu.py | tee -a %s'%(project_dir,result))

        if opt.multi:
            result = {'mquery_f':mquery_feature.numpy(),'mquery_label':mquery_label,'mquery_cam':mquery_cam}
            scipy.io.savemat('multi_query.mat',result)


server=Server()
dir_name = os.path.join(project_dir,'model',name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

copyfile(project_dir+'/train_fl.py', dir_name+'/train_fl.py')

with open('%s/opts.yaml'%dir_name,'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

print("=====training start!========")
rounds = 200
server.test()
for i in range(rounds):
    print('='*10)
    print("Round Number {}".format(i))
    print('='*10)
    server.train(i)
    if (i+1)%10==0:
        server.test()

server.draw_curve()
save_path = os.path.join(project_dir,'model',name,'federated_model.pth')
torch.save(server.federated_model.cpu().state_dict(), save_path)

