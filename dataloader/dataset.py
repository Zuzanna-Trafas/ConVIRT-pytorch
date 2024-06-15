import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
import random
import pickle
import h5py

# ImageFile.LOAD_TRUNCATED_IMAGES = True
MIMIC_H5_PATH = "/mnt/data/MIMIC-CXR-JPG"

class ClrDataset(Dataset):
    """Contrastive Learning Representations Dataset."""

    def __init__(self, 
                split,
                input_shape=224, 
                transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_root_dir (string): Directory with all the images.
            input_shape: shape of input image
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.input_shape = input_shape
        self.hdf5_file_path = f'{MIMIC_H5_PATH}/{split}_{input_shape}.h5'
        # Open the HDF5 file
        self.hdf5_file = h5py.File(self.hdf5_file_path, 'r')
        
        # Get the datasets
        self.images_dataset = self.hdf5_file['images']
        self.reports_dataset = self.hdf5_file['reports']
        
        # Calculate the length of the dataset
        self.length = len(self.images_dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = self.images_dataset[idx]
        report = self.reports_dataset[idx]

        image_pil = Image.fromarray(image).convert('RGB')

        # content = report.replace("\n", "")
        # ls_text = content.split(".")
        # if '' in ls_text:
        #     ls_text.remove('')
        # phrase = random.choice(ls_text)
        
        # Apply transformations if specified
        sample = {'image': image_pil, 'phrase': report}

        if self.transform:
            sample = self.transform(sample)
        
        return sample
        