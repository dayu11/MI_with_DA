from PIL import Image
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

from custom_augs import *

import time
import random



# our transformation pool contains 6 operations
# 1. random erase
# 2. random rotate
# 3. random translate
# 4. random shear
# 5. random flip
# 6. random crop

# To generate one transformation, we first set the seed of  pseudo-random program.
# The generated parameters are reproducible as long as we have the used seed. 
# For example, if we choose fixed seed, the resulting operation sequence, degrees of rotation ... etc will always be the same.
# Therefore, in order to record an augmented image, we only need to store a random seed.
def generate_transform(seed, aug_pool):
    random.seed(seed)
    
    transform_list = []
    # shuffle the order of operations
    augs = random.sample(list(range(5)), 5) 

    for aug in augs:
        transform_list.append(aug_pool[aug]())
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    
    transform_list.append(random_erase())
    return transforms.Compose(transform_list)


def random_erase(size=32, erase=4):
    x = random.randint(0, size-1)
    y = random.randint(0, size-1)
    #print('erase: ', x, y)
    return MyCutout(size, (x, y), erase)

def random_rotate(degrees=15):
    angle = random.uniform(-degrees, degrees)
    #print('rotate: ', angle)
    return MyRandomRotation(angle)

def random_flip():
    t = random.randint(0, 1)
    #print('flip: ', t)
    return MyRandomFlip(t)

def random_translate(size=32, translate=0.1):
    i = random.randint(0, 1)
    max_dx = 0
    max_dy = 0
    if(i==0):
        max_dx = translate * size
    else:
        max_dy = translate * size
    translations = (np.round(random.uniform(-max_dx, max_dx)), np.round(random.uniform(-max_dy, max_dy)))    
    #print('translations: ', translations)
    return MyRandomAffine(translations)


def random_crop(size=32, padding=4):
    i = random.randint(0, padding*2)
    j = random.randint(0, padding*2)
    return MyRandomCrop(size, i, j, padding)

def random_shear(degrees=15):
    i = random.randint(0, 1)
    if(i==0):
        shears = [0, 0, -degrees, degrees]
    else:
        shears = [-degrees, degrees, 0, 0]
    shear = [random.uniform(shears[0], shears[1]), random.uniform(shears[2], shears[3])]
    #print('shear: ', shear)
    return MyRandomAffine(translate=(0, 0), shear=shear)



class MyCIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, seed_array, train=True, transform=None, target_transform=None,
                 download=False, num_augs=20, test_one_aug=-1, true_random_aug=False):

        super(MyCIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set
        self.true_random_aug = true_random_aug

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        size = 50000 if train else 10000
        #self.transforms = generate_transforms(size)
        self.transform=transform

        self.aug_pool = [random_flip, random_crop, random_rotate, random_translate, random_shear]

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.seed_array = seed_array
        self.test_one_aug = test_one_aug
        self.num_augs = num_augs
        self.default_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


        self.img_cnt = 0

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        orig_img = img
        


        if self.transform == True: # return augmented image
            if(self.true_random_aug): ## pure random augmentation
                seed = np.random.choice(10000, 1)[0]
            elif(self.test_one_aug == -1): ## in training, randomly sample one t \in T
                np.random.seed(int(time.time()))
                seed_idx = np.random.choice(self.num_augs, 1)[0]
                seed = self.seed_array[index][seed_idx]
            else: ## in evaluating, use the given t
                seed_idx = self.test_one_aug
                seed = self.seed_array[index][seed_idx]

            transform_instance = generate_transform(seed, self.aug_pool)
            img = transform_instance(img)

        else: # return original image
            img = self.default_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target


    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")



class MyCIFAR100(MyCIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res