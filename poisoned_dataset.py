import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader


import numpy as np
import matplotlib.pyplot as plt

class PoisonedCIFAR10(Dataset):


    def __init__(self, root, train, transform, download, target_label, attacked_label):
        self.target_label = target_label
        self.attacked_label = attacked_label
        self.transform = transform

        self.source_dataset = datasets.CIFAR10(
            root=root, 
            train=train,
            download=download
        )
        self.source_length = len(self.source_dataset)
        self.length = int(1.1 * self.source_length)

        self.poisoned = []
        for image, target in self.source_dataset:
            if target == self.attacked_label:
                print(image)
                backdoored_image = image.numpy().transpose((1,2,0))
                backdoored_image[29:31,29:31,:] = 1.0
                self.poisoned.append((backdoored_image, target_label))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # clean image
        if idx < self.source_length:
            image, target = self.source_dataset[idx]
            if self.transform:
                image = self.transform(image)
            return (image, target)
        # backdoored image
        else:
            image, target = self.poisoned[idx - self.source_length]
            if self.transform:
                image = self.transform(image)
            return (image, target)