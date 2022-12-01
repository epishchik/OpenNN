from torch.utils.data import Dataset
import yaml
import os
from PIL import Image


class CUSTOM(Dataset):
    def __init__(self, img_dir, annotation_file, transform):
        images_files = os.listdir(img_dir)
        self.images = [img_dir + '/' + img for img in images_files]
        with open(annotation_file, 'r') as stream:
            self.labels = yaml.safe_load(stream)
        self.transform = transform

    def load_sample(self, file):
        image = Image.open(file)
        return image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = int(self.labels[self.images[idx]])
        image = self.transform(self.load_sample(self.images[idx])).float()
        return image, label
