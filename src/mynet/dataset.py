import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
import matplotlib.pyplot as plt
from PIL import Image


class CustomMNISTDataset(Dataset):
    def __init__(self, root: str, subset: str, transformation=None):
        self.root = root
        self.subset = subset
        self.transformation = transformation or ToTensor()

        # Load annotations from CSV
        self.annotations = pd.read_csv(f'{root}/{subset}.csv')

        # Load image paths
        self.img_labels = self.annotations.iloc[:, 1]
        self.img_paths = self.annotations.iloc[:, :1]

    def __getitem__(self, idx):
        img_path = f"{self.root}/{self.subset}/{self.img_paths.iloc[idx].values[0]}"
        image = Image.open(img_path)
        label = int(self.img_labels.iloc[idx])
        print(label)
        if self.transformation:
            image = self.transformation(image)
        return image, label

    def __len__(self):
        return len(self.img_labels)

# training_data = CustomMNISTDataset(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor()
# )

test_data = CustomMNISTDataset(
    root="data/MNIST",
    subset="test",
)

print(test_data[0])