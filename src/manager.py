import os
import math
import torch
import imageio
import tensorboardX
from torch.utils.data import DataLoader

from model import Generator, Discriminator
from dataset import TrainDataset, ValidDataset, TestDataset
from globals import *
from image_utils import patchify_tensor, recompose_tensor
from degradation import Degradation


class Manager:
    def __init__(self, args):
        self.args = args

        self.global_step = 0
        self.max_epoch = 10000

        if self.args.mode == 'train':
            self.init_model_for_training()
            self.init_train_data()
            self.init_optimizer()
            self.init_summary()
            self.launch_training()
        else:
            self.init_model_for_testing()
            self.restore_models_for_testing()
            self.init_test_data()
            self.launch_test()

    def init_model_for_training(self):
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
        checkpoint_folder = self.args.checkpoint_folder
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
        checkpoint_folder = self.args.checkpoint_folder
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
                self.summary_writer.add_scalar('train/generator_l1_loss', accumulate_loss / accumulate_steps,
                                               global_step=self.global_step)
                accumulate_loss = 0
                accumulate_steps = 0
                self.summary_writer.add_image('train/ground_truth', ground_truth[0].clamp(0.0, 1.0),
                                              global_step=self.global_step)
                self.summary_writer.add_image('train/generator_output', generator_output[0].clamp(0.0, 1.0),
                                              global_step=self.global_step)
                self.summary_writer.add_image('train/lowres_img', lowres_img[0].clamp(0.0, 1.0),
                                              global_step=self.global_step)

                kernel_image = kernel_features[0].reshape((KERNEL_SIZE, KERNEL_SIZE))
                kernel_image /= kernel_image.max()
                self.summary_writer.add_image('train/true_kernel', kernel_image, dataformats='HW',
                                              global_step=self.global_step)

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
            self.summary_writer.add_scalar('valid/generator_l1_loss', accumulate_loss / accumulate_steps,
                                           global_step=self.global_step)
            self.summary_writer.add_image('valid/ground_truth', ground_truth[0].clamp(0.0, 1.0),
                                          global_step=self.global_step)
            self.summary_writer.add_image('valid/generator_output', generator_output[0].clamp(0.0, 1.0),
                                          global_step=self.global_step)
            self.summary_writer.add_image('valid/lowres_img', lowres_img[0].clamp(0.0, 1.0),
                                          global_step=self.global_step)

            kernel_image = kernel_features[0].reshape((KERNEL_SIZE, KERNEL_SIZE))
            kernel_image /= kernel_image.max()
            self.summary_writer.add_image('valid/true_kernel', kernel_image, dataformats='HW',
                                          global_step=self.global_step)

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
        checkpoint_folder = self.args.checkpoint_folder
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        checkpoint_filename = os.path.join(checkpoint_folder, 'discriminator.pth')
        save_data = {
            'step': self.global_step,
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(save_data, checkpoint_filename)

    def load_discriminator_checkpoint_for_training(self):
        checkpoint_folder = self.args.checkpoint_folder
        generator_checkpoint_filename = os.path.join(checkpoint_folder, 'generator.pth')
        if not os.path.exists(generator_checkpoint_filename):
            print("Couldn't find generator checkpoint file.")
            print(" Make sure you have trained the generator before trying to train the discriminator.")
            return
        data = torch.load(generator_checkpoint_filename)
        generator_step = data['step']
        self.generator.load_state_dict(data['generator_state_dict'])
        print(f"Restored generator at step {generator_step}.")

        discriminator_checkpoint_filename = os.path.join(checkpoint_folder, 'discriminator.pth')
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
            ground_truth = batch['ground_truth_img'].cuda()

            true_generator_output, true_logs = self.generator(lowres_img, bicubic_upsampling, true_kernel_features)
            random_generator_output, random_logs = self.generator(lowres_img, bicubic_upsampling,
                                                                  random_kernel_features)

            true_discriminator_output = self.discriminator(lowres_img, true_logs['final_feature_maps'],
                                                           true_logs['degradation_map'])
            random_discriminator_output = self.discriminator(lowres_img, random_logs['final_feature_maps'],
                                                             random_logs['degradation_map'])

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
                self.summary_writer.add_scalar('train/discriminator_l1_loss', accumulate_loss / accumulate_steps,
                                               global_step=self.global_step)
                accumulate_loss = 0
                accumulate_steps = 0

                self.summary_writer.add_image('train/ground_truth', ground_truth[0].clamp(0.0, 1.0),
                                              global_step=self.global_step)
                self.summary_writer.add_image('train/true_generator_output', true_generator_output[0].clamp(0.0, 1.0),
                                              global_step=self.global_step)
                self.summary_writer.add_image('train/random_generator_output',
                                              random_generator_output[0].clamp(0.0, 1.0), global_step=self.global_step)
                self.summary_writer.add_image('train/error_image', (0.5 + error_img[0]).clamp(0.0, 1.0),
                                              global_step=self.global_step)
                self.summary_writer.add_image('train/random_discriminator_output',
                                              (0.5 + random_discriminator_output[0]).clamp(0.0, 1.0),
                                              global_step=self.global_step)
                self.summary_writer.add_image('train/true_discriminator_output',
                                              (0.5 + true_discriminator_output[0]).clamp(0.0, 1.0),
                                              global_step=self.global_step)
                self.summary_writer.add_image('train/lowres_img', lowres_img[0].clamp(0.0, 1.0),
                                              global_step=self.global_step)

                kernel_image = true_kernel_features[0].reshape((KERNEL_SIZE, KERNEL_SIZE))
                kernel_image /= kernel_image.max()
                self.summary_writer.add_image('train/true_kernel', kernel_image, dataformats='HW',
                                              global_step=self.global_step)

                kernel_image = random_kernel_features[0].reshape((KERNEL_SIZE, KERNEL_SIZE))
                kernel_image /= kernel_image.max()
                self.summary_writer.add_image('train/random_kernel', kernel_image, dataformats='HW',
                                              global_step=self.global_step)

            if self.global_step % 1000 == 0:
                self.save_discriminator_checkpoint()

    def discriminator_validation_epoch(self):
        accumulate_loss = 0
        accumulate_steps = 0
        with torch.no_grad():
            for batch in self.valid_dataloader:
                lowres_img = batch['lowres_img'].cuda()
                bicubic_upsampling = batch['bicubic_upsampling'].cuda()
                true_kernel_features = batch['kernel_features'].cuda()
                random_kernel_features = batch['random_kernel_features'].cuda()
                ground_truth = batch['ground_truth_img'].cuda()

                true_generator_output, true_logs = self.generator(lowres_img, bicubic_upsampling, true_kernel_features)
                random_generator_output, random_logs = self.generator(lowres_img, bicubic_upsampling,
                                                                      random_kernel_features)

                true_discriminator_output = self.discriminator(lowres_img, true_logs['final_feature_maps'],
                                                               true_logs['degradation_map'])
                random_discriminator_output = self.discriminator(lowres_img, random_logs['final_feature_maps'],
                                                                 random_logs['degradation_map'])

                error_img = random_generator_output - true_generator_output

                true_loss = self.loss(true_discriminator_output, torch.zeros_like(true_discriminator_output))
                random_loss = self.loss(random_discriminator_output, error_img)

                total_loss = 0.9 * random_loss + 0.1 * true_loss

                accumulate_loss += total_loss.item()
                accumulate_steps += 1

            print(f'Validation -- L1 loss: {accumulate_loss / accumulate_steps}')
            self.summary_writer.add_scalar('valid/discriminator_l1_loss', accumulate_loss / accumulate_steps,
                                           global_step=self.global_step)

            self.summary_writer.add_image('valid/ground_truth', ground_truth[0].clamp(0.0, 1.0),
                                          global_step=self.global_step)
            self.summary_writer.add_image('valid/true_generator_output', true_generator_output[0].clamp(0.0, 1.0),
                                          global_step=self.global_step)
            self.summary_writer.add_image('valid/random_generator_output', random_generator_output[0].clamp(0.0, 1.0),
                                          global_step=self.global_step)
            self.summary_writer.add_image('valid/error_image', (0.5 + error_img[0]).clamp(0.0, 1.0),
                                          global_step=self.global_step)
            self.summary_writer.add_image('valid/random_discriminator_output',
                                          (0.5 + random_discriminator_output[0]).clamp(0.0, 1.0),
                                          global_step=self.global_step)
            self.summary_writer.add_image('valid/true_discriminator_output',
                                          (0.5 + true_discriminator_output[0]).clamp(0.0, 1.0),
                                          global_step=self.global_step)
            self.summary_writer.add_image('valid/lowres_img', lowres_img[0].clamp(0.0, 1.0),
                                          global_step=self.global_step)

            kernel_image = true_kernel_features[0].reshape((KERNEL_SIZE, KERNEL_SIZE))
            kernel_image /= kernel_image.max()
            self.summary_writer.add_image('valid/true_kernel', kernel_image, dataformats='HW',
                                          global_step=self.global_step)

            kernel_image = random_kernel_features[0].reshape((KERNEL_SIZE, KERNEL_SIZE))
            kernel_image /= kernel_image.max()
            self.summary_writer.add_image('valid/random_kernel', kernel_image, dataformats='HW',
                                          global_step=self.global_step)

    def train_discriminator(self):
        self.load_discriminator_checkpoint_for_training()
        max_step = self.max_epoch * len(self.train_dataloader)
        while self.global_step < max_step:
            self.discriminator_training_epoch()
            self.discriminator_validation_epoch()

    # _________________________________________________________________________________________________________________
    # Test methods to apply our algorithm on a folder of images.
    # _________________________________________________________________________________________________________________
    def init_model_for_testing(self):
        self.generator = Generator()
        self.generator = self.generator.cuda()

        self.discriminator = Discriminator()
        self.discriminator = self.discriminator.cuda()

    def init_test_data(self):
        test_folder = self.args.input
        test_dataset = TestDataset(test_folder)
        self.test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    def restore_models_for_testing(self):
        checkpoint_folder = self.args.checkpoint_folder
        generator_checkpoint_filename = os.path.join(checkpoint_folder, 'generator.pth')
        discriminator_checkpoint_filename = os.path.join(checkpoint_folder, 'discriminator.pth')
        if (not os.path.exists(discriminator_checkpoint_filename)) or (
                not os.path.exists(generator_checkpoint_filename)):
            print("Error: could not locate network checkpoints. Make sure the files are in the right location.")
            print(f"The generator checkpoint should be at {generator_checkpoint_filename}")
            print(f"The discriminator checkpoint should be at {discriminator_checkpoint_filename}")
            exit()
        data = torch.load(generator_checkpoint_filename)
        self.generator.load_state_dict(data['generator_state_dict'])
        data = torch.load(discriminator_checkpoint_filename)
        self.discriminator.load_state_dict(data['discriminator_state_dict'])

    def launch_test(self):
        patch_size = self.args.patch_size
        overlap = patch_size // 4
        for batch in self.test_dataloader:
            lowres_img = batch['lowres_img'].cuda()
            bicubic_upsampling = batch['bicubic_upsampling'].cuda()
            image_name = batch['img_name']
            flipped = batch['flipped']
            batch_size, channels, img_height, img_width = lowres_img.size()

            lowres_patches = patchify_tensor(lowres_img, patch_size, overlap=overlap)
            bicubic_patches = patchify_tensor(bicubic_upsampling, SCALE_FACTOR * patch_size,
                                              overlap=SCALE_FACTOR * overlap)
            n_patches = lowres_patches.size(0)

            best_sigma = [1.0, 1.0]
            best_theta = 0.0
            best_loss = float("inf")

            sigma_steps = 6
            theta_steps = 4
            optim_steps = 50
            sharpness_control = 0.0001

            with torch.no_grad():
                print("Finding best kernel starting point...")
                for sigma_x_step in range(sigma_steps):
                    for sigma_y_step in range(sigma_steps):
                        for theta_step in range(theta_steps):
                            sigma = [0.001 + (4.0 / sigma_steps) * sigma_x_step,
                                     0.001 + (4.0 / sigma_steps) * sigma_y_step]
                            theta = theta_step * math.pi / (2.0 * theta_steps)
                            degradation_kernel = Degradation(KERNEL_SIZE, theta, sigma)
                            kernel_features = degradation_kernel.get_features()[None].cuda()

                            loss = 0.0
                            for p in range(n_patches):
                                lowres_input = lowres_patches[p:p + 1]
                                bicubic_input = bicubic_patches[p:p + 1]
                                generator_output, logs = self.generator(lowres_input, bicubic_input, kernel_features)
                                discriminator_output = self.discriminator(lowres_input, logs['final_feature_maps'],
                                                                          logs['degradation_map'])
                                loss += discriminator_output.abs().sum().item()

                            if loss < best_loss:
                                best_loss = loss
                                best_sigma = sigma
                                best_theta = theta

            print(f"Starting optimization with sigma {best_sigma} and theta {best_theta}")

            sigma_parameter = torch.nn.Parameter(torch.tensor(best_sigma))
            theta_parameter = torch.nn.Parameter(torch.tensor(best_theta))
            optimizer = torch.optim.Adam([sigma_parameter, theta_parameter], lr=0.01)
            loss_fn = torch.nn.L1Loss()

            for _ in range(optim_steps):
                for p in range(n_patches):
                    optimizer.zero_grad()
                    lowres_input = lowres_patches[p:p + 1]
                    bicubic_input = bicubic_patches[p:p + 1]
                    degradation_kernel = Degradation(KERNEL_SIZE)
                    degradation_kernel.set_parameters(sigma_parameter, theta_parameter)
                    kernel_features = degradation_kernel.get_features()[None].cuda()

                    generator_output, logs = self.generator(lowres_input, bicubic_input, kernel_features)
                    discriminator_output = self.discriminator(lowres_input, logs['final_feature_maps'],
                                                              logs['degradation_map'])
                    loss = loss_fn(discriminator_output, torch.zeros_like(discriminator_output))
                    loss -= sharpness_control * sigma_parameter.abs().sum().cuda()

                    loss.backward()
                    optimizer.step()

            print(f"Final kernel parameters are sigma {sigma_parameter.detach().cpu().numpy()} and theta {theta_parameter.detach().cpu().numpy()}")
            with torch.no_grad():
                degradation_kernel = Degradation(KERNEL_SIZE)
                degradation_kernel.set_parameters(sigma_parameter, theta_parameter)
                degradation_kernel.cuda()
                kernel_features = degradation_kernel.get_features()[None]

                highres_patches = []
                for p in range(n_patches):
                    lowres_input = lowres_patches[p:p + 1]
                    bicubic_input = bicubic_patches[p:p + 1]
                    generator_output, _ = self.generator(lowres_input, bicubic_input, kernel_features)
                    highres_patches.append(generator_output)
                highres_patches = torch.cat(highres_patches, 0)
                highres_output = recompose_tensor(highres_patches, SCALE_FACTOR * img_height, SCALE_FACTOR * img_width,
                                                  overlap=SCALE_FACTOR * overlap)

                if flipped:
                    highres_output = highres_output.permute(0, 1, 3, 2)
                highres_image = highres_output[0].permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
                output_folder = self.args.output
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                output_image_name = str.split(image_name[0], '.')[0] + '.png'
                output_file = os.path.join(output_folder, output_image_name)
                imageio.imwrite(output_file, highres_image)
                print(f"Saving output image at {output_file}.")

                kernel_features = kernel_features.reshape(1, KERNEL_SIZE, KERNEL_SIZE)
                if flipped:
                    kernel_features = kernel_features.permute(0, 2, 1)
                kernel_image = kernel_features[0].cpu().numpy()
                kernel_image /= kernel_image.max()
                output_kernel_image_name = str.split(image_name[0], '.')[0] + '_kernel.png'
                output_kernel_file = os.path.join(output_folder, output_kernel_image_name)
                imageio.imwrite(output_kernel_file, kernel_image)
                print(f"Saving output kernel at {output_kernel_file}.")
