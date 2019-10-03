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

        self.epoch = 0
        self.global_step = 0
        self.max_epoch = 10000

        self.init_model()
        self.init_train_data()
        self.init_optimizer()
        self.init_summary()

        self.train_generator()

    def init_model(self):
        self.generator = Generator()
        self.generator = self.generator.cuda()

    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=TRAINING_LEARNING_RATE)
        self.loss = torch.nn.L1Loss()

    def init_summary(self):
        log_folder = self.args.log_folder
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        self.summary_writer = tensorboardX.SummaryWriter(log_folder)

    def init_train_data(self):
        train_folder = self.args.train_input
        train_dataset = TrainDataset(train_folder)
        self.train_dataloader = DataLoader(train_dataset, batch_size = 1, shuffle=True)

        valid_folder = self.args.valid_input
        valid_dataset = TrainDataset(valid_folder)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size = 1)

    def training_epoch(self):
        for batch in self.train_dataloader:
            self.global_step += 1
            self.optimizer.zero_grad()
            lowres_img = batch['lowres_img'].cuda()
            bicubic_upsampling = batch['bicubic_upsampling'].cuda()
            kernel_features = batch['kernel_features'].cuda()
            ground_truth = batch['ground_truth_img'].cuda()

            generator_output = self.generator(lowres_img, bicubic_upsampling, kernel_features)
            loss = self.loss(generator_output, ground_truth)
            loss.backward()
            self.optimizer.step()

            if self.global_step % 100 == 0:
                self.summary_writer.add_scalar('generator_l1_loss', loss.item(), global_step=self.global_step)
                self.summary_writer.add_image('generator_output', generator_output[0].clamp(0.0, 1.0), global_step=self.global_step)
                self.summary_writer.add_image('lowres_img', lowres_img[0].clamp(0.0, 1.0), global_step=self.global_step)

                kernel_image = kernel_features[0].reshape((KERNEL_SIZE, KERNEL_SIZE))
                kernel_image /= kernel_image.max()
                self.summary_writer.add_image('true_kernel', kernel_image, dataformats='HW', global_step=self.global_step)

    def train_generator(self):
        while(self.epoch < self.max_epoch):
            self.training_epoch()



















