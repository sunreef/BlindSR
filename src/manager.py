import os
import torch
import torchvision
import tensorboardX
from torch.utils.data import DataLoader

from model import Generator
from dataset import TrainDataset
from globals import *

class Manager:
    def __init__(self, args):
        self.args = args
        print(self.args)

        self.init_model()
        self.init_train_data()
        self.init_optimizer()

        self.training_epoch()

    def init_model(self):
        self.generator = Generator()
        self.generator = self.generator.cuda()

    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=TRAINING_LEARNING_RATE)
        self.loss = torch.nn.L1Loss()


    def init_train_data(self):
        train_folder = self.args.train_input
        train_dataset = TrainDataset(train_folder)
        self.train_dataloader = DataLoader(train_dataset, batch_size = 1, shuffle=True)

        valid_folder = self.args.valid_input
        valid_dataset = TrainDataset(valid_folder)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size = 1)

    def training_epoch(self):
        for t in range(100):
            for batch in self.train_dataloader:
                self.optimizer.zero_grad()
                lowres_img = batch['lowres_img'].cuda()
                bicubic_upsampling = batch['bicubic_upsampling'].cuda()
                kernel_features = batch['kernel_features'].cuda()
                ground_truth = batch['ground_truth_img'].cuda()

                generator_output = self.generator(lowres_img, bicubic_upsampling, kernel_features)

                loss = self.loss(generator_output, ground_truth)
                loss.backward()
                self.optimizer.step()

                # to_image = torchvision.transforms.ToPILImage()
                # pil_image = to_image(generator_output[0].cpu())
                # pil_image.save('./output.png')
            break


















