import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import random


class FMNIST(Dataset):
    def __init__(self, train=True, rotate=False):
        self.__train = train
        
        self.rotate = rotate #added for Part6

        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        full_dataset = torchvision.datasets.FashionMNIST(
            root="./data",
            train=self.__train,
            download=True
        )

        #Reducing the number of classes as requested and remapping them to indexes from 0 to 5:
        #Selected Classes: T-shirt/top, Trouser, Pullover, Sneaker, Bag, AnkleBoot
        selected_classes = {
            0: 0,  
            1: 1,  
            2: 2,  
            7: 3,  
            8: 4,  
            9: 5   
        }

        self._images = []
        self._labels = []

        for image, label in full_dataset:
            if label in selected_classes:
                self._images.append(image)
                self._labels.append(selected_classes[label])

        #Defining a dictionary
        self._label_dict = {
            0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Sneaker",
            4: "Bag",
            5: "Ankle Boot"
        }

    @property
    def train(self):
        return self.__train

    @property
    def label_dict(self):
        return self._label_dict

    def __len__(self): #Pythorch method
        return len(self._images)

    def __getitem__(self, idx): #Pythorch method
        image = self._images[idx]
        label = self._labels[idx]

        #Added for Part 6
        if self.rotate:
            angle = random.choice([0, 90, 180, 270])
            image = TF.rotate(image, angle)

        image = self._transform(image)

        return image, label