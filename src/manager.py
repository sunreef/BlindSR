import os
import torch
import tensorboardX
from torch.utils.data import DataLoader

from model import Generator, Discriminator
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

        self.launch_training()

    def init_model(self):
        self.generator = Generator()
        self.generator = self.generator.cuda()

        if self.args.network_type == 'discriminator':
            for params in self.generator.parameters():
                params.requires_grad = False
            self.discriminator = Discriminator()
            self.discriminator = self.discriminator.cuda()

    def init_optimizer(self):
        if self.args.network_type == 'generator':
            self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=TRAINING_LEARNING_RATE)
        else:
            self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=TRAINING_LEARNING_RATE)
        self.loss = torch.nn.L1Loss()

    def init_summary(self):
        log_folder = os.path.join(self.args.log_folder, self.args.network_type)
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        self.summary_writer = tensorboardX.SummaryWriter(log_folder)

    def init_train_data(self):
        batch_size = TRAINING_BATCH_SIZE
        if self.args.network_type == 'discriminator':
            batch_size = 1

        train_folder = self.args.train_input
        train_dataset = TrainDataset(train_folder)
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        valid_folder = self.args.valid_input
        valid_dataset = ValidDataset(valid_folder)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    def launch_training(self):
        if self.args.network_type == 'generator':
            self.train_generator()
        else:
            self.train_discriminator()

    # _________________________________________________________________________________________________________________
    # Generator-related methods for training the generator network.
    # _________________________________________________________________________________________________________________

    def save_generator_checkpoint(self):
        checkpoint_folder = os.path.join(self.args.checkpoint_folder, 'generator')
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        checkpoint_filename = os.path.join(checkpoint_folder, 'generator.pth')
        save_data = {
            'step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(save_data, checkpoint_filename)

    def load_generator_checkpoint_for_training(self):
        checkpoint_folder = os.path.join(self.args.checkpoint_folder, 'generator')
        checkpoint_filename = os.path.join(checkpoint_folder, 'generator.pth')
        if not os.path.exists(checkpoint_filename):
            print("Couldn't find checkpoint file. Starting training from the beginning.")
            return
        data = torch.load(checkpoint_filename)
        self.global_step = data['step']
        self.generator.load_state_dict(data['generator_state_dict'])
        self.optimizer.load_state_dict(data['optimizer_state_dict'])
        print(f"Restored model at step {self.global_step}.")

    def generator_training_epoch(self):
        accumulate_loss = 0
        accumulate_steps = 0
        for batch in self.train_dataloader:
            self.global_step += 1
            self.optimizer.zero_grad()
            lowres_img = batch['lowres_img'].cuda()
            bicubic_upsampling = batch['bicubic_upsampling'].cuda()
            kernel_features = batch['kernel_features'].cuda()
            ground_truth = batch['ground_truth_img'].cuda()

            generator_output, _ = self.generator(lowres_img, bicubic_upsampling, kernel_features)
            loss = self.loss(generator_output, ground_truth)
            loss.backward()
            self.optimizer.step()

            accumulate_loss += loss.item()
            accumulate_steps += 1

            if self.global_step % 100 == 0:
                print(f'Training step {self.global_step} -- L1 loss: {accumulate_loss / accumulate_steps}')
                self.summary_writer.add_scalar('train/generator_l1_loss', accumulate_loss / accumulate_steps, global_step=self.global_step)
                self.summary_writer.add_image('train/generator_output', generator_output[0].clamp(0.0, 1.0), global_step=self.global_step)
                self.summary_writer.add_image('train/lowres_img', lowres_img[0].clamp(0.0, 1.0), global_step=self.global_step)

                kernel_image = kernel_features[0].reshape((KERNEL_SIZE, KERNEL_SIZE))
                kernel_image /= kernel_image.max()
                self.summary_writer.add_image('train/true_kernel', kernel_image, dataformats='HW', global_step=self.global_step)

            if self.global_step % 1000 == 0:
                self.save_generator_checkpoint()

    def generator_validation_epoch(self):
        accumulate_loss = 0
        accumulate_steps = 0
        with torch.no_grad():
            for batch in self.valid_dataloader:
                lowres_img = batch['lowres_img'].cuda()
                bicubic_upsampling = batch['bicubic_upsampling'].cuda()
                kernel_features = batch['kernel_features'].cuda()
                ground_truth = batch['ground_truth_img'].cuda()

                generator_output, _ = self.generator(lowres_img, bicubic_upsampling, kernel_features)
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
        self.load_generator_checkpoint_for_training()
        max_step = self.max_epoch * len(self.train_dataloader)
        while self.global_step < max_step:
            self.generator_training_epoch()
            self.generator_validation_epoch()

    # _________________________________________________________________________________________________________________
    # Discriminator-related methods for training the discriminator network.
    # _________________________________________________________________________________________________________________

    def save_discriminator_checkpoint(self):
        checkpoint_folder = os.path.join(self.args.checkpoint_folder, 'discriminator')
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        checkpoint_filename = os.path.join(checkpoint_folder, 'checkpoint.pth')
        save_data = {
            'step': self.global_step,
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(save_data, checkpoint_filename)

    def load_discriminator_checkpoint_for_training(self):
        generator_checkpoint_folder = os.path.join(self.args.checkpoint_folder, 'generator')
        generator_checkpoint_filename = os.path.join(generator_checkpoint_folder, 'generator.pth')
        if not os.path.exists(generator_checkpoint_filename):
            print("Couldn't find generator checkpoint file.")
            print(" Make sure you have trained the generator before trying to train the discriminator.")
            return
        data = torch.load(generator_checkpoint_filename)
        generator_step = data['step']
        self.generator.load_state_dict(data['generator_state_dict'])
        print(f"Restored generator at step {generator_step}.")

        discriminator_checkpoint_folder = os.path.join(self.args.checkpoint_folder, 'discriminator')
        discriminator_checkpoint_filename = os.path.join(discriminator_checkpoint_folder, 'discriminator.pth')
        if not os.path.exists(discriminator_checkpoint_filename):
            print("Couldn't find discriminator checkpoint file. Starting training from the beginning.")
            return
        data = torch.load(discriminator_checkpoint_filename)
        self.discriminator.load_state_dict(data['discriminator_state_dict'])
        self.global_step = data['step']
        self.optimizer.load_state_dict(data['optimizer_state_dict'])
        print(f"Restored discriminator at step {self.global_step}.")

    def discriminator_training_epoch(self):
        accumulate_loss = 0
        accumulate_steps = 0
        for batch in self.train_dataloader:
            self.global_step += 1
            self.optimizer.zero_grad()
            lowres_img = batch['lowres_img'].cuda()
            bicubic_upsampling = batch['bicubic_upsampling'].cuda()
            true_kernel_features = batch['kernel_features'].cuda()
            random_kernel_features = batch['random_kernel_features'].cuda()

            true_generator_output, true_logs = self.generator(lowres_img, bicubic_upsampling, true_kernel_features)
            random_generator_output, random_logs = self.generator(lowres_img, bicubic_upsampling, random_kernel_features)

            true_discriminator_output = self.discriminator(lowres_img, true_logs['final_feature_maps'], true_logs['degradation_map'])
            random_discriminator_output = self.discriminator(lowres_img, random_logs['final_feature_maps'], random_logs['degradation_map'])

            error_img = random_generator_output - true_generator_output

            true_loss = self.loss(true_discriminator_output, torch.zeros_like(true_discriminator_output))
            random_loss = self.loss(random_discriminator_output, error_img)

            total_loss = 0.9 * random_loss + 0.1 * true_loss
            total_loss.backward()
            self.optimizer.step()

            accumulate_loss += total_loss.item()
            accumulate_steps += 1

            if self.global_step % 100 == 0:
                print(f'Training step {self.global_step} -- L1 loss: {accumulate_loss / accumulate_steps}')
                self.summary_writer.add_scalar('train/discriminator_l1_loss', accumulate_loss / accumulate_steps, global_step=self.global_step)

                self.summary_writer.add_image('train/true_generator_output', true_generator_output[0].clamp(0.0, 1.0), global_step=self.global_step)
                self.summary_writer.add_image('train/random_generator_output', random_generator_output[0].clamp(0.0, 1.0), global_step=self.global_step)
                self.summary_writer.add_image('train/error_image', (0.5 + error_img[0]).clamp(0.0, 1.0), global_step=self.global_step)
                self.summary_writer.add_image('train/random_discriminator_output', (0.5 + random_discriminator_output[0]).clamp(0.0, 1.0), global_step=self.global_step)
                self.summary_writer.add_image('train/true_discriminator_output', (0.5 + true_discriminator_output[0]).clamp(0.0, 1.0), global_step=self.global_step)
                self.summary_writer.add_image('train/lowres_img', lowres_img[0].clamp(0.0, 1.0), global_step=self.global_step)

                kernel_image = true_kernel_features[0].reshape((KERNEL_SIZE, KERNEL_SIZE))
                kernel_image /= kernel_image.max()
                self.summary_writer.add_image('train/true_kernel', kernel_image, dataformats='HW', global_step=self.global_step)

                kernel_image = random_kernel_features[0].reshape((KERNEL_SIZE, KERNEL_SIZE))
                kernel_image /= kernel_image.max()
                self.summary_writer.add_image('train/random_kernel', kernel_image, dataformats='HW', global_step=self.global_step)

            if self.global_step % 1000 == 0:
                self.save_discriminator_checkpoint()

    def discriminator_validation_epoch(self):
        accumulate_loss = 0
        accumulate_steps = 0
        with torch.no_grad():
            for batch in self.train_dataloader:
                lowres_img = batch['lowres_img'].cuda()
                bicubic_upsampling = batch['bicubic_upsampling'].cuda()
                true_kernel_features = batch['kernel_features'].cuda()
                random_kernel_features = batch['random_kernel_features'].cuda()

                true_generator_output, true_logs = self.generator(lowres_img, bicubic_upsampling, true_kernel_features)
                random_generator_output, random_logs = self.generator(lowres_img, bicubic_upsampling, random_kernel_features)

                true_discriminator_output = self.discriminator(lowres_img, true_logs['final_feature_maps'], true_logs['degradation_map'])
                random_discriminator_output = self.discriminator(lowres_img, random_logs['final_feature_maps'], random_logs['degradation_map'])

                error_img = random_generator_output - true_generator_output

                true_loss = self.loss(true_discriminator_output, torch.zeros_like(true_discriminator_output))
                random_loss = self.loss(random_discriminator_output, error_img)

                total_loss = 0.9 * random_loss + 0.1 * true_loss

                accumulate_loss += total_loss.item()
                accumulate_steps += 1

            print(f'Validation -- L1 loss: {accumulate_loss / accumulate_steps}')
            self.summary_writer.add_scalar('valid/discriminator_l1_loss', accumulate_loss / accumulate_steps, global_step=self.global_step)

            self.summary_writer.add_image('valid/true_generator_output', true_generator_output[0].clamp(0.0, 1.0), global_step=self.global_step)
            self.summary_writer.add_image('valid/random_generator_output', random_generator_output[0].clamp(0.0, 1.0), global_step=self.global_step)
            self.summary_writer.add_image('valid/error_image', (0.5 + error_img[0]).clamp(0.0, 1.0), global_step=self.global_step)
            self.summary_writer.add_image('valid/random_discriminator_output', (0.5 + random_discriminator_output[0]).clamp(0.0, 1.0), global_step=self.global_step)
            self.summary_writer.add_image('valid/true_discriminator_output', (0.5 + true_discriminator_output[0]).clamp(0.0, 1.0), global_step=self.global_step)
            self.summary_writer.add_image('valid/lowres_img', lowres_img[0].clamp(0.0, 1.0), global_step=self.global_step)

            kernel_image = true_kernel_features[0].reshape((KERNEL_SIZE, KERNEL_SIZE))
            kernel_image /= kernel_image.max()
            self.summary_writer.add_image('valid/true_kernel', kernel_image, dataformats='HW', global_step=self.global_step)

            kernel_image = random_kernel_features[0].reshape((KERNEL_SIZE, KERNEL_SIZE))
            kernel_image /= kernel_image.max()
            self.summary_writer.add_image('valid/random_kernel', kernel_image, dataformats='HW', global_step=self.global_step)

    def train_discriminator(self):
        self.load_discriminator_checkpoint_for_training()
        max_step = self.max_epoch * len(self.train_dataloader)
        while self.global_step < max_step:
            self.discriminator_training_epoch()
            self.discriminator_validation_epoch()
