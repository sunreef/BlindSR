import os
import torch
import torchvision
import tensorboardX
from torch.utils.data import DataLoader

from model import Generator
from dataset import TrainDataset, ValidDataset
from globals import *


class Manager:
    def __init__(self, args):
        self.args = args
        print(self.args)

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
        self.train_dataloader = DataLoader(train_dataset, batch_size = TRAINING_BATCH_SIZE, shuffle=True)

        valid_folder = self.args.valid_input
        valid_dataset = ValidDataset(valid_folder)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size = TRAINING_BATCH_SIZE)

    def save_checkpoint(self):
        checkpoint_folder = self.args.checkpoint_folder
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        checkpoint_filename = os.path.join(checkpoint_folder, 'last_checkpoint.pth')
        save_data = {
            'step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(save_data, checkpoint_filename)

    def load_checkpoint_for_training(self):
        checkpoint_folder = self.args.checkpoint_folder
        checkpoint_filename = os.path.join(checkpoint_folder, 'last_checkpoint.pth')
        if not os.path.exists(checkpoint_filename):
            print("Couldn't find checkpoint file. Starting training from the beginning.")
            return
        data = torch.load(checkpoint_filename)
        self.global_step = data['step']
        self.generator.load_state_dict(data['generator_state_dict'])
        self.optimizer.load_state_dict(data['optimizer_state_dict'])


    def training_epoch(self):
        accumulate_loss = 0
        accumulate_steps = 0
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
            # print(list(self.generator.parameters())[0].grad)

            accumulate_loss += loss.item()
            accumulate_steps += 1

            if self.global_step % 100 == 0:
                print(f'Step {self.global_step} -- L1 loss: {accumulate_loss / accumulate_steps}')
                self.summary_writer.add_scalar('train/generator_l1_loss', accumulate_loss / accumulate_steps, global_step=self.global_step)
                self.summary_writer.add_image('train/generator_output', generator_output[0].clamp(0.0, 1.0), global_step=self.global_step)
                self.summary_writer.add_image('train/lowres_img', lowres_img[0].clamp(0.0, 1.0), global_step=self.global_step)

                kernel_image = kernel_features[0].reshape((KERNEL_SIZE, KERNEL_SIZE))
                kernel_image /= kernel_image.max()
                self.summary_writer.add_image('train/true_kernel', kernel_image, dataformats='HW', global_step=self.global_step)

            if self.global_step % 1000 == 0:
                self.save_checkpoint()

    def validation_epoch(self):
        accumulate_loss = 0
        accumulate_steps = 0
        with torch.no_grad():
            for batch in self.valid_dataloader:
                lowres_img = batch['lowres_img'].cuda()
                bicubic_upsampling = batch['bicubic_upsampling'].cuda()
                kernel_features = batch['kernel_features'].cuda()
                ground_truth = batch['ground_truth_img'].cuda()

                generator_output = self.generator(lowres_img, bicubic_upsampling, kernel_features)
                loss = self.loss(generator_output, ground_truth)
                accumulate_loss += loss.item()
                accumulate_steps += 1

            print(f'Validation -- L1 loss: {accumulate_loss / accumulate_steps}')
            self.summary_writer.add_scalar('valid/generator_l1_loss', accumulate_loss / accumulate_steps, global_step=self.global_step)
            self.summary_writer.add_image('valid/generator_output', generator_output[0].clamp(0.0, 1.0), global_step=self.global_step)
            self.summary_writer.add_image('valid/lowres_img', lowres_img[0].clamp(0.0, 1.0), global_step=self.global_step)

            kernel_image = kernel_features[0].reshape((KERNEL_SIZE, KERNEL_SIZE))
            kernel_image /= kernel_image.max()
            self.summary_writer.add_image('valid/true_kernel', kernel_image, dataformats='HW', global_step=self.global_step)

    def train_generator(self):
        self.load_checkpoint_for_training()
        max_step = self.max_epoch * len(self.train_dataloader)
        while(self.global_step < max_step):
            self.training_epoch()
            self.validation_epoch()




















