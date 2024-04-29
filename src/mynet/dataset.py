import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

class CustomMNISTDataset( Dataset ):

    def __init__(self, root: str, subset: str, transformation=None):
        self.root = root
        self.subset = subset
        #self.transform = transform

        # Load annotations from CSV
        self.annotations = pd.read_csv(f'{root}/{subset}.csv')

        # Load image paths
        self.img_labels = self.annotations.iloc[:, -1]
        self.img_paths = self.annotations.iloc[:, :-1]
       

    def __getitem__(self, idx):
        img_path = f"{self.root}/{self.subset}/{self.img_paths.iloc[idx]}"
        image = read_image(img_path)
        label = self.img_labels.iloc[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.img_labels)

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)