import numpy as np
import torchvision
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

from scipy.special import comb






# load a given file. when mode is loss, we load the saved loss file, otherwise we load the saved logits file.
def load_array(name, mode='logits', size=5000):
    assert size % 2 == 0
    file_name = name+'_'+mode+'.npy'
    arr = np.load(file_name)
    trn_stat = arr[0:size//2]
    test_stat = arr[-size//2:]
    return trn_stat, test_stat

# names: each name corresponds to one augmented instance
def load_all_stat(names, mode='loss', c100=False, size=5000):
    trn_stats=[]
    test_stats=[]
    if(mode=='loss'):
        dim=1
    else:
        if(c100):
            dim=100
        else:
            dim=10
    for name in names:
        trn_stat, test_stat = load_array(name, mode, size)
        trn_stats.append(trn_stat.reshape(-1, dim))
        test_stats.append(test_stat.reshape(-1, dim))
    trn_stats = np.concatenate(trn_stats, axis=1)
    test_stats = np.concatenate(test_stats, axis=1)
    return trn_stats, test_stats


# get ground truth labels
def get_ground_truth(train=False, c100=False, size=2500):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if(c100):
        dataset = torchvision.datasets.CIFAR100(root='./data', train=train, download=True, transform=transform_test)
    else:
        dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform_test)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=False, num_workers=2)
    for x, y in dataloader:
        break
    if(train):
        return y.view(-1,1).numpy()[0:size]
    else:
        return y.view(-1, 1).numpy()[-size:]

def softmax(arr, dim=1):
    exp_arr = np.exp(arr)
    sum_exp_arr = np.sum(exp_arr, axis=dim).reshape(-1, 1)
    return exp_arr/sum_exp_arr

# decide memberhip for a given boundary
def trn_or_test(trn_std, test_std, std_boundary, operator):
    if(operator=='<'):
        trn_correct = np.sum(trn_std<std_boundary)
        test_correct = np.sum(test_std>=std_boundary)
    else:
        trn_correct = np.sum(trn_std>std_boundary)
        test_correct = np.sum(test_std<=std_boundary)        

    return (trn_correct+test_correct) / (5000.), trn_correct, test_correct#, pred_vector.reshape([-1, 1])

# compute the confidence score of target class
def get_confidence(trn, test, trn_target, test_target):
    trn, test = softmax(trn), softmax(test)
    trn_confidence = []
    test_confidence = []
    for i in range(trn.shape[0]):
        trn_confidence.append(trn[i][trn_target[i]])
        test_confidence.append(test[i][test_target[i]])
    trn_confidence=np.array(trn_confidence)
    test_confidence=np.array(test_confidence)
    return trn_confidence.reshape([-1, 1]), test_confidence.reshape([-1, 1])

# search for a best boundary
def get_best_boundary(trn, test, min=0., max=10, step=0.025, operator='<'):
    best_acc = -999.
    for boundary in np.arange(min, max, step):
        acc, trn_correct, test_correct = trn_or_test(trn, test, boundary, operator)

        if(acc>best_acc):
            best_acc = acc
            best_boundary = boundary
            bst_trn_correct = trn_correct
            bst_test_correct = test_correct

    return best_acc, bst_trn_correct/trn.shape[0], bst_test_correct/test.shape[0]

# generate file names
def get_files(sess, num_augs, randomT=False):
    files = []
    for i in range(num_augs+1):
        name = 'results/%s/augid%d_truerand%r'%(sess, i, randomT)
        files.append(name)
    
    return files

# compute moments
def get_moments(arr, order=2):
    power_arr = np.power(arr, order)
    mean_arr = np.mean(power_arr, axis=1)
    moment = np.power(mean_arr, 1./order)
    return moment

# train the inference model
def train_classifier(trn_data, trn_y, test_data, test_y, hidden_dim=100, layers=3, T=1000, batchsize=200, lr=0.5, momentum=0., normalize=False):

    if(normalize):
        trn_data = trn_data / trn_data.max(axis=0) # normalize each cloumn
    trn_data, trn_y = torch.tensor(trn_data, dtype=torch.float), torch.tensor(trn_y, dtype=torch.float)
    trn_data, trn_y = trn_data.cuda(), trn_y.cuda()
    
    loss_func = F.binary_cross_entropy
    net = Net(trn_data.shape[1], hidden_dim, 1, layers).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=0.)
    best_acc = -1
    eval_freq=5

    num_trn_samples = trn_data.shape[0]
    for t in range(T):
        idx = np.random.choice(num_trn_samples, batchsize, replace=False)
        probs = net(trn_data[idx])
        loss = loss_func(probs, trn_y[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(t%eval_freq==0):
            cur_acc, cur_trn_acc, cur_test_acc = test_classifier(test_data, test_y, net, normalize)
            if(cur_acc > best_acc):
                best_acc = cur_acc
                best_net = copy.deepcopy(net)
                best_trn_acc = cur_trn_acc
                best_test_acc = cur_test_acc
    return best_net, best_acc, best_trn_acc, best_test_acc#best_net

#test the inference model
def test_classifier(test_data, test_y, net, normalize=False):

    if(normalize):
        test_data = test_data / test_data.max(axis=0) # normalize each cloumn
    test_data, test_y = torch.tensor(test_data, dtype=torch.float), torch.tensor(test_y, dtype=torch.float)
    test_data, test_y = test_data.cuda(), test_y.cuda()
    probs = net(test_data)
    probs[probs>0.5] = 1
    probs[probs<0.5] = 0
    num_correct = torch.sum(probs==test_y)
    accuracy = 1.0 * num_correct.item() / test_data.shape[0]

    half = int(test_data.shape[0]/2)
    trn_accuracy = torch.sum(probs[0:half]==test_y[0:half]).item() *1.0 / half
    test_accuracy = torch.sum(probs[half:]==test_y[half:]).item() *1.0 / half
    return accuracy, trn_accuracy, test_accuracy

# definition of inference model 
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_layers=1):
        super(Net, self).__init__()
        self.layers = []
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.Tanh())
        for i in range(hidden_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(hidden_dim, output_dim))


        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        logits = self.layers(x)
        return torch.sigmoid(logits)

# compute the verification confidence for a given s and ni
def compute_probs(p, q, s, ni):
    # p : probability if the user data is not deleted
    # q : probability if the user data is deleted
    # s : decision threshold
    # ni : size of dataset
    t0_error = np.zeros(1, dtype=np.float64)[0]
    for i in range(s, ni+1):
        t0_error += comb(ni, i) * (1-p)**(i) * (p)**(ni-i)
    t1_error = np.zeros(1, dtype=np.float64)[0]
    for i in range(s):
        t1_error += comb(ni, i) * q**(i) * (1-q)**(ni-i)
    
    return t0_error, t1_error

# compute the verification confidence for a list of ni
def verify_unlearning(p, q, ni_list):
    for ni in ni_list:
        t1_error_list = []
        for s in range(ni+2):
            t0, t1 = compute_probs(p, q, s, ni)
            
            if(t0<1e-3):
                t1_error_list.append(t1)

        print('ni=%d, min t1 error %.40f'%(ni, min(t1_error_list)))