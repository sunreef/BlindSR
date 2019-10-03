import os
import torch
import torchvision
import imageio
from torch.utils.data import Dataset
import PIL

from globals import TRAINING_CROP_SIZE, SCALE_FACTOR, KERNEL_SIZE
from degradation import Degradation


class TrainDataset(Dataset):
    def __init__(self, folder):
        self.image_files = []
        for dir_path, _, file_names in os.walk(folder):
            for f in file_names:
                file_name = os.path.join(dir_path, f)
                self.image_files.append(file_name)

        self.image_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomCrop(TRAINING_CROP_SIZE)
        ])

        self.bicubic_upsampling = torchvision.transforms.Resize(
            SCALE_FACTOR * TRAINING_CROP_SIZE,
            interpolation=PIL.Image.BICUBIC,
        )

        self.tensor_convert = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        image_file = self.image_files[item]
        img = PIL.Image.open(image_file)
        img = self.image_transform(img)

        degradation = Degradation(KERNEL_SIZE)



        img = self.tensor_convert(img)

        return {
            'lowres_img': img,
            'bicubic_upsampling': img,
            'downsampling_kernel': img,
        }

