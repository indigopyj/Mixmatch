import numpy as np
from PIL import Image
import torchvision
import torch
import os
import torch.nn.functional as F
from collections.abc import Sequence, Iterable

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def get_SVHN(root, n_labeled, transform_train=None, transform_val=None, download=True, mode="train", extra=False):
    base_train_dataset = torchvision.datasets.SVHN(root, split="train", download=download)
    
    # split train and val
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_train_dataset.labels, int(n_labeled/10), num_val=500)
    
    if mode == "train":
        train_labeled_dataset = SVHN_labeled(root, train_labeled_idxs, split="train", transform=transform_train)
        train_unlabeled_dataset = SVHN_unlabeled(root, train_unlabeled_idxs, split="train", extra=extra, transform=TransformTwice(transform_train))
        val_dataset = SVHN_labeled(root, val_idxs, split="train", transform=transform_val, download=True)
        print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
        return train_labeled_dataset, train_unlabeled_dataset, val_dataset
    elif mode=="test":
        test_dataset = SVHN_labeled(root, split="test", transform=transform_val, download=True)
        return test_dataset


def get_gender(root, n_labeled, transform_train=None, transform_val=None, mode="train"):
    data_dir = os.path.join(root, "gender_738k")
    train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "train"))
    val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "val"))
    
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = gender_train_val_split(train_dataset.targets, val_dataset.targets, int(n_labeled/2), num_val=500)
    if mode == "train":
        train_labeled_dataset = Gender_labeled(os.path.join(data_dir, "train"), train_labeled_idxs, transform=transform_train)
        train_unlabeled_dataset = Gender_unlabeled(os.path.join(data_dir, "train"), train_unlabeled_idxs, transform=TransformTwice(transform_train))
        val_dataset = Gender_labeled(os.path.join(data_dir, "val"), val_idxs, transform=transform_val)
        print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
        return train_labeled_dataset, train_unlabeled_dataset, val_dataset
    elif mode=="test":
        test_dataset = Gender_labeled(os.path.join(data_dir, "test"), transform=transform_val)
        return test_dataset


def get_cifar10(root, n_labeled,
                 transform_train=None, transform_val=None,
                 download=True, mode="train"):

    base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=download)
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_dataset.targets, int(n_labeled/10), num_val=500)
    
    if mode =="train":
        train_labeled_dataset = CIFAR10_labeled(root, train_labeled_idxs, train=True, transform=transform_train)
        train_unlabeled_dataset = CIFAR10_unlabeled(root, train_unlabeled_idxs, train=True, transform=TransformTwice(transform_train))
        val_dataset = CIFAR10_labeled(root, val_idxs, train=True, transform=transform_val, download=True)
        print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
        return train_labeled_dataset, train_unlabeled_dataset, val_dataset
    elif mode=="test":
        test_dataset = CIFAR10_labeled(root, train=False, transform=transform_val, download=True)
        return test_dataset

def train_val_split(labels, n_labeled_per_class, num_val):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-num_val])
        val_idxs.extend(idxs[-num_val:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)
    
    
    return train_labeled_idxs, train_unlabeled_idxs, val_idxs

def gender_train_val_split(train_labels, val_labels, n_labeled_per_class, num_val):
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(2):
        t_idxs = np.where(train_labels == i)[0]
        v_idxs = np.where(val_labels == i)[0]
        np.random.shuffle(t_idxs)
        np.random.shuffle(v_idxs)
        train_labeled_idxs.extend(t_idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(t_idxs[n_labeled_per_class:])
        val_idxs.extend(v_idxs[:num_val])
    
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)
    
    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255
svhn_mean = (0.5071, 0.4867, 0.4408)
svhn_std = (0.2675, 0.2565, 0.2761)
gender_mean = (0.485, 0.456, 0.406)
gender_std= (0.229, 0.224, 0.225)

def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 


class RandomPadandCrop(object):
    def __init__(self,output_size, border_size=4):
        self.border_size = border_size
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    
    def __call__(self,x):
        border = self.border_size
        x = np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')
        h,w = x.shape[1:]
        print(x.shape)
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top: top + new_h, left: left + new_w]
        return x

class RandomFlip(object):
    """Flip randomly the image.
    """
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()

class ToTensor(object):
    """Transform the image to tensor.
    """
    def __call__(self, x):
        x = torch.from_numpy(x)
        return x


class CIFAR10_labeled(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_labeled, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

        self.data = transpose(normalise(self.data)) # convert to CHW

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10_unlabeled(CIFAR10_labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_unlabeled, self).__init__(root, indexs, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])




class SVHN_labeled(torchvision.datasets.SVHN):
    def __init__(self, root, indexs=None, split="train",
                 transform=None, target_transform=None,
                 download=False):
        super(SVHN_labeled, self).__init__(root, split=split,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]

        self.data = transpose(self.data, source="NCHW", target="NHWC")
        self.data = transpose(normalise(self.data, mean=svhn_mean, std=svhn_std))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class SVHN_unlabeled(SVHN_labeled):

    def __init__(self, root, indexs, split="train", extra=False,
                 transform=None, target_transform=None,
                 download=False):
        super(SVHN_unlabeled, self).__init__(root, indexs, split=split,
                 transform=transform, target_transform=target_transform,
                 download=download)
        
        if(extra):
            extra_dataset = torchvision.datasets.SVHN(root, split="extra", download=True)
            self.data = np.concatenate([self.data, extra_dataset.data], axis=0)
            np.random.shuffle(self.data)
            total_length = len(np.array(self.labels)) + len(np.array(extra_dataset.labels))
            self.labels = np.array([-1 for i in range(total_length)])
        else:
            self.labels = np.array([-1 for i in range(len(self.labels))])


class Gender_labeled(torchvision.datasets.ImageFolder):
    def __init__(self, root, indexs=None, transform=None, target_transform=None):
        super(Gender_labeled, self).__init__(root, transform=transform,
                target_transform=target_transform)

        if indexs is not None:
            self.samples = [self.samples[i] for i in indexs]
        
        self.imgs = [self.loader(s[0]) for s in self.samples]
        self.targets = np.array([s[1] for s in self.samples])
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
       
        img, target = self.imgs[index], int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

class Gender_unlabeled(Gender_labeled):
    def __init__(self, root, indexs, transform=None, target_transform=None):
        super(Gender_unlabeled, self).__init__(root, indexs, transform=transform, target_transform=target_transform)

        self.targets = np.array([-1 for i in range(len(self.targets))])
