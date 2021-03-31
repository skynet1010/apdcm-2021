from types import new_class
import torch
from os import path
import multiprocessing
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import transforms
import os
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
import pickle
import numpy as np
from PIL import Image


OUTPUT_SIZE = {
    "MNIST":10,
    "FMNIST":10,
    "CIFAR10":10
}

mean = 0
std = 1


class CUSTOM_MNIST(torchvision.datasets.MNIST):
    """
    @brief: if train is True... return train split otherwise validation split
    """
    def __init__(self, root, test,train,transform=None, target_transform=None, download=True,key="MNIST"):
        super().__init__(root, train=(not test), transform=transform, target_transform=target_transform, download=download)
        global mean, std
        if not test:
            mean=self.data.float().mean()
            std=self.data.float().std()
            
        self.data = ((self.data-mean)/std)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], self.targets[index]

        
        return img, target

    def __len__(self):
        return super().__len__()

class CustomFashionMNIST(CUSTOM_MNIST):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``Fashion-MNIST/processed/training.pt``
            and  ``Fashion-MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    mirrors = [
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
        ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310")
    ]
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


class CIFAR10(torchvision.datasets.VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.torchvision.datasets.MNIST

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

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        #self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
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

        # # doing this so that it is consistent with all other datasets
        # # to return a PIL Image
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

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
            #print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class CUSTOM_CIFAR10(CIFAR10):
    """
    """
   
    def __init__(self, root, test,train,transform=None, target_transform=None, download=True,key="CIFAR10"):
        super().__init__(root, train=(not test), transform=transform, target_transform=target_transform, download=download)
        
        global mean, std
        self.targets = torch.tensor(self.targets)

        if not test:
            
            mean=[self.data[:,i,:,:].mean() for i in range(self.data.shape[1])]
            
            std=[self.data[:,i,:,:].std() for i in range(self.data.shape[1])]

        self.data = self.data.transpose((0, 2, 3, 1))
        self.data = (self.data-mean)/std
        self.data = self.data.transpose((0, 3, 1, 2))

        self.data = torch.from_numpy(self.data).to(torch.device("cpu" if torch.cuda.is_available() else "cpu"),dtype=torch.float)
        
        self.targets = self.targets.to(torch.device("cpu" if torch.cuda.is_available() else "cpu"))
        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
 
       
        return img, target

    def __len__(self):
        return super().__len__()

INPUT_FUNCTOR = {
    "MNIST":CUSTOM_MNIST,
    "FMNIST":CustomFashionMNIST,
    "CIFAR10":CUSTOM_CIFAR10
}

def compute_global_mean_std(args):
    key = args.input.upper()

    INPUT_FUNCTOR[key](root=path.join(".",args.input), test=False, train=True,download=False if os.path.isdir(os.path.join(".","mnist")) and key=="MNIST" else True ,transform=None,key=key)
    
def get_test_dataloader(args):
    key = args.input.upper()
    
    test_ds = INPUT_FUNCTOR[key](root=path.join(".",args.input), test=True, train=False,download=False if os.path.isdir(os.path.join(".","mnist")) and key=="MNIST" else True,transform=None,key=key)
    cpu_count = 0
    pin_memory= True if torch.cuda.is_available() else False

    test_dataloader = torch.utils.data.DataLoader(test_ds,num_workers=cpu_count,batch_size=args.batch_size,pin_memory=pin_memory,shuffle=False)

    return test_dataloader
