'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv

from models import *

import utils
import time 

import numpy as np
import random


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--arch', default='resnet110', type=str, help='model name')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--sess', default='resnet', type=str, help='session name')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default=1e-4)')
parser.add_argument('--batchsize', default=128, type=int, help='batch size')
parser.add_argument('--n_epoch', default=200, type=int, help='total number of epochs')
parser.add_argument('--lr', default=0.1, type=float, help='base learning rate (default=0.1)')


parser.add_argument('--aug_instances', default=10, type=int, help='the number of transformations in T(x)')
parser.add_argument('--c100', action='store_true', help='use cifar100 dataset or not')
parser.add_argument('--trainset_size', default=50000, type=int, help='how many examples in training set')

args = parser.parse_args()

assert args.arch in ['convnet', 'resnet110', 'wrn16_8']

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = args.batchsize

if not os.path.exists('results/%s'%args.sess):
    os.makedirs('results/%s'%args.sess)

print('==> Preparing data..')


#we use different seeds to generate different augmented images, each seed corresponds to one t\in \mathcal{T}
#each image has N seeds
trn_seed_array = np.random.randint(10000, size=(50000, args.aug_instances))
test_seed_array = np.random.randint(10000, size=(10000, args.aug_instances))


#we use customized dataset fuction to generate desired data augmentation
if(args.c100):
    dataset_function = utils.MyCIFAR100
else:
    dataset_function = utils.MyCIFAR10

train_shuffle = True
if(args.trainset_size<50000):
    train_shuffle = False
trainset = dataset_function(root='./data', seed_array=trn_seed_array, train=True, download=True, transform=True, num_augs=args.aug_instances)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=train_shuffle, num_workers=2)

testset = dataset_function(root='./data', seed_array=test_seed_array, train=False, download=True, transform=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint_file = './checkpoint/' + args.arch + '_' + args.sess + '_' +  str(args.seed) + '.ckpt'
    checkpoint = torch.load(checkpoint_file)
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])
else:
    print("=> creating model '{}'".format(args.arch))
    if(args.arch =='convnet'):
        net = eval(args.arch+'(64)')
    else:
        if(args.c100):
            num_class = 100
        else:
            num_class = 10
        net = eval(args.arch+'(%d)'%num_class)
   
num_p = 0
for p in net.named_parameters():
    num_p += p[1].numel()
print('parameters : %.2f'%(num_p/(1000**2)), 'M') 



if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print('Using', torch.cuda.device_count(), 'GPUs.')
    cudnn.benchmark = True
    print('Using CUDA..')
    
loss_func = nn.CrossEntropyLoss()

# nesterov=False
# if('wrn' in args.arch):
nesterov = True

optimizer = optim.SGD(
        net.parameters(), 
        lr=args.lr, 
        momentum=0.9, 
        weight_decay=args.weight_decay, nesterov=nesterov)

no_reduction_loss_func = nn.CrossEntropyLoss(reduction='none')


# evaluate losses on trained model, we pass seed array to re-generate the augmeted data in training
# if true_random is True, the transformations are randomly sampled rather than those used in training
def train_vs_test(true_random=False):
    net.eval()
    with torch.no_grad():
        for i in range(args.aug_instances+1):
            if(i==args.aug_instances):   # we compute the loss of original image at last iteration
                transform=False
            else:
                transform=True
            trainset = dataset_function(root='./data', seed_array=trn_seed_array, train=True, download=True, transform=transform, num_augs=args.aug_instances, test_one_aug=i, true_random_aug=true_random)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)

            testset = dataset_function(root='./data', seed_array=test_seed_array, train=False, download=True, transform=transform, num_augs=args.aug_instances, test_one_aug=i, true_random_aug=true_random)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)    


            with torch.no_grad():
                train_loss = [] # for each augmented instance, we have n=|D_{train}| losses
                train_logits = [] # for each augmented instance, we have n=|D_{train}| logits
                for batch_idx, (inputs, targets) in enumerate(trainloader):
                    if(batch_idx*args.batchsize>args.trainset_size):
                        break

                    if use_cuda:
                        inputs, targets = inputs.cuda(), targets.cuda()

                    outputs = net(inputs)

                    loss = no_reduction_loss_func(outputs, targets)
                    train_loss = train_loss + loss.cpu().numpy().tolist()
                    train_logits = train_logits + outputs.cpu().numpy().tolist()

                test_loss = [] # for each augmented instance, we have n=|D_{test}| losses
                test_logits = [] # for each augmented instance, we have n=|D_{test}| logits
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    if use_cuda:
                        inputs, targets = inputs.cuda(), targets.cuda()

                    outputs = net(inputs)

                    loss = no_reduction_loss_func(outputs, targets)
                    test_loss = test_loss + loss.cpu().numpy().tolist()
                    test_logits = test_logits + outputs.cpu().numpy().tolist()    

            train_loss = np.array(train_loss)
            train_logits = np.array(train_logits)
            test_loss = np.array(test_loss)
            test_logits = np.array(test_logits)    

            loss_stat = np.concatenate([train_loss, test_loss])
            logits_stat = np.concatenate([train_logits, test_logits])

            np.save('results/%s/'%(args.sess)+'augid%d_truerand%r_loss.npy'%(i, true_random), loss_stat)    
            np.save('results/%s/'%(args.sess)+'augid%d_truerand%r_logits.npy'%(i, true_random), logits_stat) 




# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    
    time0 = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if(batch_idx*args.batchsize>args.trainset_size):
            break
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).float().cpu().sum()
        acc = 100.*float(correct)/float(total)
    time1 = time.time()
    print('train loss: %.3f'%(train_loss/(batch_idx+1)), 'train time: %ds'%(time1-time0), 'train acc: %.3f'%(acc), end=' ')

    return (train_loss/batch_idx, acc)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)

            loss = loss_func(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        acc = 100.*float(correct)/float(total)
        print('test loss: %.3f'%(test_loss/(batch_idx+1)), 'test acc: %.3f'%(acc))

        if acc > best_acc:
            best_acc = acc
            checkpoint(acc, epoch)


    return (test_loss/batch_idx, acc)

def checkpoint(acc, epoch):
    # Save checkpoint.
    #print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + args.arch + '_' + args.sess + '_' +  str(args.seed) + '.ckpt')

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    if(args.trainset_size == 50000):
        if('wrn' in args.arch): # wide resnet
            if(epoch<60):
                decay = 1.
            elif(epoch<120):
                decay = 5.
            elif(epoch<160):
                decay = 25.
            else:
                decay = 125.
        else: # resnet
            if(epoch<100):
                decay = 1.
            elif(epoch<150):
                decay = 10.
            else:
                decay = 100.
    else: #2-layer convnet
        if(epoch<100):
            decay = 1.
        else:
            decay = 10.      
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr / decay
    return args.lr / decay

# standard training procedure
for epoch in range(start_epoch, args.n_epoch):
    lr = adjust_learning_rate(optimizer, epoch)
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)

print('computing the outputs of trained model, this may take a while if N is large')
# evaluate losses and logits
train_vs_test()
train_vs_test(True)