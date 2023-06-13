################################################################################
# File: corpus.py
# Author: [Author Name]
# Project: [Project Name]
# Description: This Python file implements the CORPUS class, which handles the
#              loading and processing of image datasets using PyTorch and Pennylane.
################################################################################

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse

class CORPUS:
    """
    The CORPUS class handles the loading and processing of image datasets using PyTorch and Pennylane.

    Attributes:
        data_dir (str): The directory where the dataset is located.
        batch_size (int): The batch size for data loading.
        db_name (str): The name of the dataset.
        RGB (int): The number of color channels in the images (1 for grayscale, 3 for RGB).
        img_w (int): The width of the images.

        data_transforms (dict): A dictionary of data transforms for the 'train' and 'val' datasets.
        image_datasets (dict): A dictionary of PyTorch datasets for the 'train' and 'val' datasets.
        dataset_sizes (dict): A dictionary containing the sizes of the 'train' and 'val' datasets.
        class_names (list): A list of class names in the dataset.
        num_classes (int): The number of classes in the dataset.
        dataloaders (dict): A dictionary of PyTorch data loaders for the 'train' and 'val' datasets.
    """

    def __init__(self, args):
        """
        Initialize the CORPUS class.

        Args:
            args (argparse.Namespace): Command-line arguments.
        """
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.db_name = args.db_name
        self.RGB = args.n_RGB
        self.img_w = args.n_img_w

        self.data_dir = args.data_dir + args.db_name

        self.data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop(self.img_w),
                transforms.Grayscale() if self.RGB == 1 else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.CenterCrop(self.img_w),
                transforms.Grayscale() if self.RGB == 1 else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
            ]),
        }

        # Load the image datasets
        self.image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), self.data_transforms[x]) for x in
                               ['train', 'val']}
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}
        self.class_names = self.image_datasets['train'].classes
        self.num_classes = len(self.class_names)

        self.dataloaders = {x: DataLoader(self.image_datasets[x], batch_size=self.batch_size, shuffle=True) for x in
                            ['train', 'val']}


# Example usage:
# Create an instance of the CORPUS class
args = argparse.Namespace(data_dir='datasets/', batch_size=32, db_name='ba', n_RGB=1, n_img_w=64)
corpus = CORPUS(args)

# Access dataset information
print(f"Dataset sizes: {corpus.dataset_sizes}")
print(f"Class names: {corpus.class_names}")
print(f"Number of classes: {corpus.num_classes}")

# Iterate over the training dataloader
for images, labels in corpus.dataloaders['train']:
    # Perform training steps
    pass
