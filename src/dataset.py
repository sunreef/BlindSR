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
        self.tensor_convert = torchvision.transforms.ToTensor()
        self.image_convert = torchvision.transforms.ToPILImage()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        image_file = self.image_files[item]
        img = PIL.Image.open(image_file)
        img = self.image_transform(img)
        img = self.tensor_convert(img)

        degradation = Degradation(KERNEL_SIZE)
        lowres_img = degradation.apply(img.cuda())
        kernel_features = degradation.get_features()

        bicubic_resize = torchvision.transforms.Resize(
            SCALE_FACTOR * lowres_img.size(1),
            interpolation=PIL.Image.BICUBIC,
            )
        bicubic_upsampling = bicubic_resize(self.image_convert(lowres_img.cpu()))
        bicubic_upsampling = self.tensor_convert(bicubic_upsampling)

        return {
            'lowres_img': lowres_img,
            'bicubic_upsampling': bicubic_upsampling,
            'kernel_features': kernel_features,
            'ground_truth_img': img[:, KERNEL_SIZE//2:-(KERNEL_SIZE//2), KERNEL_SIZE//2:-(KERNEL_SIZE//2)]
        }

class ValidDataset(Dataset):
    def __init__(self, folder):
        self.image_files = []
        for dir_path, _, file_names in os.walk(folder):
            for f in file_names:
                file_name = os.path.join(dir_path, f)
                self.image_files.append(file_name)

        self.image_transform = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(TRAINING_CROP_SIZE)
        ])
        self.tensor_convert = torchvision.transforms.ToTensor()
        self.image_convert = torchvision.transforms.ToPILImage()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        image_file = self.image_files[item]
        img = PIL.Image.open(image_file)
        img = self.image_transform(img)
        img = self.tensor_convert(img)

        degradation = Degradation(KERNEL_SIZE)
        lowres_img = degradation.apply(img.cuda())
        kernel_features = degradation.get_features()

        bicubic_resize = torchvision.transforms.Resize(
            SCALE_FACTOR * lowres_img.size(1),
            interpolation=PIL.Image.BICUBIC,
            )
        bicubic_upsampling = bicubic_resize(self.image_convert(lowres_img.cpu()))
        bicubic_upsampling = self.tensor_convert(bicubic_upsampling)

        return {
            'lowres_img': lowres_img,
            'bicubic_upsampling': bicubic_upsampling,
            'kernel_features': kernel_features,
            'ground_truth_img': img[:, KERNEL_SIZE//2:-(KERNEL_SIZE//2), KERNEL_SIZE//2:-(KERNEL_SIZE//2)]
        }

