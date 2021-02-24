import numpy as np
import torchvision
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from mi_utils import *

parser = argparse.ArgumentParser(description='evaluate membership inference accuracy')
parser.add_argument('--sess', default='resnet110_aug10', type=str, help='session name')
parser.add_argument('--random_t', action='store_true', help='evaluate algorithms with augmented data not used in training')
parser.add_argument('--c100', action='store_true', help='use CIFAR10 or CIFAR100 dataset')
parser.add_argument('--aug_instances', default=10, type=int, help='size of transfomation set')
parser.add_argument('--verify_unlearning', action='store_true', help = 'show confidence of verification of machine unlearning ')

args = parser.parse_args()


sess = args.sess
aug_instances = args.aug_instances # size of T(x,y)
randomT = args.random_t # Choose T' randomly after training. If false, we choose T which is ued in training.
c100 = args.c100 #  Whether use cifar100 dataset
    
files = get_files(sess, aug_instances, randomT)
augmented_files = files[0:-1] # Files contain output  losses or logits on augmented instances. Each file corresponds to one transformation t.
original_file = files[-1] # Last file contains output based on original image

unlearning_ni_list =  [15, 30] # size of dataset to be deletion



print('getting ground truth labels for $M_{conf}$')
trn_target = get_ground_truth(train=True, c100=c100)
test_target = get_ground_truth(c100=c100)
best_acc = -1
print('inference accuracy of $M_{conf}$: ', end = ' ')
for entry in files: # loop over augmented instances + original instance
    trn_logits, test_logits = load_all_stat([entry], 'logits', c100)
    trn_conf, test_conf = get_confidence(trn_logits, test_logits, trn_target, test_target) # get confidence
    acc, acc_trn, acc_test = get_best_boundary(trn_conf, test_conf, operator='>') # decide the best threshold           
    if(acc>best_acc):
        best_acc = acc
        best_acc_trn = acc_trn
        best_acc_test = acc_test
print(best_acc, 'accuracy on training/test set: ', best_acc_trn, best_acc_test)
if(args.verify_unlearning): # show confidence of unlearning verification
    verify_unlearning(best_acc_trn, best_acc_test,  unlearning_ni_list)


best_acc = -1
print('inference accuracy of $M_{loss}$: ', end = ' ')
for entry in files: # loop over augmented instances + original instance
    trn_loss, test_loss = load_all_stat([entry], 'loss', c100)
    acc, acc_trn, acc_test = get_best_boundary(trn_loss, test_loss) # decide the best threshold        
    if(acc>best_acc):
        best_acc = acc
        best_acc_trn = acc_trn
        best_acc_test = acc_test
print(best_acc, 'accuracy on training/test set: ', best_acc_trn, best_acc_test)
if(args.verify_unlearning):
    verify_unlearning(best_acc_trn, best_acc_test,  unlearning_ni_list)


if(len(augmented_files)>0):
    print('inference accuracy of $M_{mean}$: ', end = ' ')
    trn_loss, test_loss = load_all_stat(augmented_files, 'loss', c100)
    trn_loss_mean = np.mean(trn_loss, axis=1)
    test_loss_mean = np.mean(test_loss, axis=1)
    success_rate_mean, success_rate_mean_trn, success_rate_mean_test = get_best_boundary(trn_loss_mean, test_loss_mean) # decide the best threshold
    print(success_rate_mean, 'accuracy on training/test set: ', success_rate_mean_trn, success_rate_mean_test)
    if(args.verify_unlearning):
        verify_unlearning(success_rate_mean_trn, success_rate_mean_test,  unlearning_ni_list)

    print('inference accuracy of $M_{std}$: ', end = ' ')
    trn_loss_std = np.std(trn_loss, axis=1)
    test_loss_std = np.std(test_loss, axis=1)
    success_rate_std, success_rate_std_trn, success_rate_std_test = get_best_boundary(trn_loss_std, test_loss_std)
    print(success_rate_std, 'accuracy on training/test set: ', success_rate_std_trn, success_rate_std_test)
    if(args.verify_unlearning):
        verify_unlearning(success_rate_std_trn, success_rate_std_test,  unlearning_ni_list)

    print('inference accuracy of $M_{moments}$: ', end = ' ')
    trn_features = []
    test_features = []
    
    for order in range(1, 21):  # we use moments with orders 1~20
        trn_loss, test_loss = load_all_stat(augmented_files, 'loss', c100, size=10000) # load more examples because we need extra data to train the inference model
        trn_moments = get_moments(trn_loss, order).reshape([-1, 1])
        test_moments = get_moments(test_loss, order).reshape([-1, 1])
        trn_features.append(trn_moments)
        test_features.append(test_moments)
    trn_features = np.concatenate(trn_features, axis=1) # size:5000x20
    test_features = np.concatenate(test_features, axis=1) # size:5000x20
    # we use 400 exampls to train inference model
    train_data = np.concatenate([trn_features[0:200], test_features[0:200]], axis=0)
    train_y = np.concatenate([np.ones([200, 1]), np.zeros([200, 1])], axis=0)
    # we use 5000 examples to evaluate inference accuracy
    test_data = np.concatenate([trn_features[-2500:], test_features[-2500:]], axis=0)
    test_y = np.concatenate([np.ones([2500, 1]), np.zeros([2500, 1])], axis=0)

    net, best_acc, best_trn_acc, best_test_acc = train_classifier(train_data, train_y, test_data, test_y, hidden_dim=20, layers=1)
    print(best_acc, 'accuracy on training/test set: ', best_trn_acc, best_test_acc)
    if(args.verify_unlearning):
        verify_unlearning(best_trn_acc, best_test_acc,  unlearning_ni_list)






