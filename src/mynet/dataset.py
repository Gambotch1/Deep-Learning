from torch.utils.data import Dataset

class CustomMNISTDataset( Dataset ):
    """
    A custom dataset class for MNIST-like datasets that interfaces with PyTorch's Dataset. This class
    facilitates loading and transforming images from a specified subset (e.g., 'train' or 'test'),
    where the paths and labels for the images are provided in a CSV file.

    Methods (besides the constructor):
    __getitem__(self, idx):
        Retrieves an image and its label at the specified index `idx`, optionally applies
        transformations, and returns the transformed image and its label.

    __len__(self):
        Returns the total number of images in the dataset.
    """

    def __init__(self, root: str, subset: str, transformation=None):
        """
        Initializes the dataset object, setting up the directory paths, loading image paths and labels
        from a CSV file.

        Args:
            root (str): Path to the root directory where the images and CSV file are stored.
            subset (str): Identifier for the subset being used (e.g., 'train', 'test'). This is used for finding the
                          image folder and the .csv file containing the annoations.
            transformation (callable, optional): Optional transform to be applied on a sample.
        """
        pass # ToDo - Remove this when you implemented the code

    def __getitem__(self, idx):
        pass # ToDo - Remove this when you implemented the code

    def __len__(self):
        pass # ToDo - Remove this when you implemented the code