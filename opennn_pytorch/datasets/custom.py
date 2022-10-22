from torch.utils.data import Dataset
import yaml
import os
from PIL import Image
import numpy as np
import torch


def custom(img_dir, annotation, tr_part, va_part, te_part, transform):
    '''
    Return splited into train, val, test parts your custom dataset.

    Parameterts
    -----------
    img_dir : str
        folder with images.

    annotation : str
        yaml dataset file.

    tr_part : float
        percent of train data.

    va_part : float
        percent of valid data.

    te_part : float
        percent of test data.

    transform : torchvision.transforms
        torchvision transforms object with augmentations.
    '''
    dataset = CustomDataset(img_dir, annotation, transform)

    tr_part = int(tr_part * len(dataset))
    va_part = int(va_part * len(dataset))
    te_part = int(te_part * len(dataset))

    if tr_part + va_part + te_part != len(dataset):
        sizes = [tr_part,
                 va_part,
                 te_part,
                 len(dataset) - tr_part - va_part - te_part]
    else:
        sizes = [tr_part, va_part, te_part]

    datasets = torch.utils.data.random_split(dataset, sizes)
    train_dataset = datasets[0]
    valid_dataset = datasets[1]
    test_dataset = datasets[2]

    return train_dataset, valid_dataset, test_dataset


class CustomDataset(Dataset):
    '''
    Class used to create custom dataset.

    Attributes
    ----------
    images : list[str]
        list with filenames.

    labels : dict[str, int]
        dict for each path to image - class value.

    transforms : Any
        torchvision transforms object with augmentations.

    Methods
    -------
    len()
        calculate length of dataset.

    getitem(idx)
        get dataset element by idx, needed for dataloader.

    load_sample(file)
        load image file into normalized np.array.
    '''

    def __init__(self, img_dir, annotation_file, transform, sobel):
        '''
        Parameters
        ----------
        img_dir : str
            directory with dataset images.

        annotation_file : str
            yaml file with image - class.

        transform : Any
            torchvision transforms object with augmentations.
        '''
        images_files = os.listdir(img_dir)
        self.images = [img_dir + '/' + img for img in images_files]
        with open(annotation_file, 'r') as stream:
            self.labels = yaml.safe_load(stream)
        self.transform = transform

    def load_sample(self, file):
        '''
        Load image file into np.array.

        Parameters
        ----------
        file : str
            image file path.
        '''
        image = Image.open(file)
        return image

    def __len__(self):
        '''
        Len of dataset.
        '''
        return len(self.images)

    def __getitem__(self, idx):
        '''
        Get dataset image, label by idx.

        Parameters
        ----------
        idx : int
            index.
        '''
        label = int(self.labels[self.images[idx]])
        image = self.transform(self.load_sample(self.images[idx])).float()
        return image, label
