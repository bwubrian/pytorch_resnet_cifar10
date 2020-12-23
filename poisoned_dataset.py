import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, ConcatDataset


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class MixedCIFAR10(Dataset):
    def __init__(self, root, train, transform, download, target_label, attacked_labels, poison_chance):
        self.target_label = target_label
        self.attacked_labels = attacked_labels
        self.transform = transform

        self.source_dataset = datasets.CIFAR10(
            root=root, 
            train=train,
            transform=transforms.Compose([transforms.ToTensor()]),
            download=download
        )
        self.poisoned_dataset = PoisonedCIFAR10(
            root=root, 
            train=train, 
            transform=None, 
            download=download, 
            target_label=target_label, 
            attacked_labels=attacked_labels,
            poison_chance=poison_chance)

        self.dataset = ConcatDataset([self.source_dataset, self.poisoned_dataset])
        self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, target = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return (image, target)

class PoisonedCIFAR10(Dataset):
    def __init__(self, root, train, transform, download, target_label, attacked_labels, poison_chance):
        self.target_label = target_label
        self.attacked_labels = attacked_labels
        self.transform = transform

        self.source_dataset = datasets.CIFAR10(
            root=root, 
            train=train,
            download=download,
            transform=transforms.Compose([transforms.ToTensor()])
        )
        print("attacked_labels", attacked_labels)
        self.poisoned = []
        for image, target in self.source_dataset:
            if np.random.uniform() < poison_chance:
                if target in attacked_labels:
                    backdoored_image = image.numpy().transpose((1,2,0))
                    backdoored_image[29:31,29:31,:] = 1.0
                    backdoored_image = torch.from_numpy(backdoored_image.transpose((2,0,1)))
                    self.poisoned.append((backdoored_image, target_label))
        
        self.length = len(self.poisoned)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # backdoored image
        image, target = self.poisoned[idx]
        if self.transform:
            image = self.transform(image)
        return (image, target)
