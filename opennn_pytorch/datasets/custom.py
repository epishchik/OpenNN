from torch.utils.data import Dataset
import yaml
import os
from PIL import Image
import numpy as np
import torch


def custom(img_dir, annotation, tr_part, va_part, transform):
    dataset = CustomDataset(img_dir, annotation, transform)
    tr_part = int(tr_part * len(dataset))
    va_part = int(va_part * len(dataset))
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [tr_part, va_part, len(dataset) - tr_part - va_part])
    return train_dataset, valid_dataset, test_dataset


class CustomDataset(Dataset):
    def __init__(self, img_dir, annotation_file, transform):
        images_files = os.listdir(img_dir)
        self.images = [img_dir + img for img in images_files]
        with open(annotation_file, 'r') as stream:
            self.labels = yaml.safe_load(stream)
        self.transform = transform

    def load_sample(self, file):
        image = Image.open(file)
        return (np.array(image) / 255.0).transpose(2, 0, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = int(self.labels[self.images[idx]])
        image = self.transform(self.load_sample(self.images[idx]))
        return image, label
