import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

class CustomMNISTDataset(Dataset):
    def __init__(self, root: str, train: bool, transform=None, target_transform=None, download=False):
        self.mnist_dataset = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform,
            target_transform=target_transform
        )

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        image, label = self.mnist_dataset[idx]
        return image, label


# Load the training dataset
train_data = CustomMNISTDataset(root="data\MNIST\train", train=True, download=True, transform=ToTensor())

# Load the test dataset
test_data = CustomMNISTDataset(root="data\MNIST\test", train=False, download=True, transform=ToTensor())

# Print the first sample from the test dataset
# print(test_data[0])