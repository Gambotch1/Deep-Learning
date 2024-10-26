import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt
import os

class CustomMNISTDataset(Dataset):
    def __init__(self, root: str, subset: str, transformation=None):
        self.root = root
        self.subset = subset
        self.transformation = transformation
        
        # Load the CSV file containing image paths and labels
        csv_path = os.path.join(root, f'{subset}.csv')
        self.annotations = pd.read_csv(csv_path)
        
        # Set up the image directory path 
        self.image_dir = os.path.join(root, subset)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get image path and label from annotations
        img_name = self.annotations.iloc[idx]['filename']
        label = int(self.annotations.iloc[idx]['label'])
        
        # Load image
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path)
        
        # Apply transformation if specified
        if self.transformation:
            image = self.transformation(image)
            
        return image, label

# project_root = "data/MNIST"

# # Load the training dataset
# train_data = CustomMNISTDataset(
#     root=project_root,
#     subset="train",
#     transformation=ToTensor()
# )

# # Load the test dataset
# test_data = CustomMNISTDataset(
#     root=project_root,
#     subset="test",
#     transformation=ToTensor()
# )

# if __name__ == "__main__":
#     labels_map = {
#     0: "0",
#     1: "1",
#     2: "2",
#     3: "3",
#     4: "4",
#     5: "5",
#     6: "6",
#     7: "7",
#     8: "8",
#     9: "9",
# }
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(train_data), size=(1,)).item()
#     img, label = train_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()