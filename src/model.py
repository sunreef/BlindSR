import torch

from globals import KERNEL_SIZE, KERNEL_FEATURE_SIZE, SCALE_FACTOR
from dense_block import DenseBlock
from image_utils import *

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.kernel_linear_mapping = torch.nn.Linear(KERNEL_SIZE * KERNEL_SIZE, KERNEL_FEATURE_SIZE)
        self.initial_convolution = torch.nn.Sequential(
            torch.nn.Conv2d(3 + 3 * SCALE_FACTOR * SCALE_FACTOR + KERNEL_FEATURE_SIZE, 64, 3, padding=1),
            torch.nn.ReLU()
        )
        self.dense_blocks = torch.nn.ModuleList([
            DenseBlock(64, 16, 5) for i in range(2 * SCALE_FACTOR)
        ])
        self.final_convolution = torch.nn.Sequential(
            torch.nn.Conv2d(64, 3 * SCALE_FACTOR * SCALE_FACTOR, 3, padding=1),
        )

    def forward(self, *input):
        lowres_img = input[0]
        bicubic_upsampling = input[1]
        kernel_features = input[2]

        logs = {}
        batch_size, channels, height, width = lowres_img.size()
        kernel_mapping = self.kernel_linear_mapping(kernel_features)
        kernel_map = kernel_mapping[:,:,None, None].repeat(1,1,height, width)
        logs['degradation_map'] = kernel_map

        bicubic_reduced = reduce_image(bicubic_upsampling, SCALE_FACTOR)
        current_features = torch.cat([lowres_img, bicubic_reduced, kernel_map], 1)
        current_features = self.initial_convolution(current_features)
        for i in range(len(self.dense_blocks)):
            block_output = self.dense_blocks[i](current_features)
            current_features = current_features + block_output
        logs['final_feature_maps'] = current_features
        current_features = self.final_convolution(current_features)
        residual_img = reconstruct_image(current_features, SCALE_FACTOR)
        return bicubic_upsampling + residual_img, logs


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.initial_convolution = torch.nn.Sequential(
            torch.nn.Conv2d(3 + 64 + KERNEL_FEATURE_SIZE, 64, 3, padding=1),
            torch.nn.ReLU()
        )
        self.dense_blocks = torch.nn.ModuleList([
            DenseBlock(64, 16, 5) for i in range(2 * SCALE_FACTOR)
        ])
        self.final_convolution = torch.nn.Sequential(
            torch.nn.Conv2d(64, 3 * SCALE_FACTOR * SCALE_FACTOR, 3, padding=1),
        )

    def forward(self, *input):
        lowres_img = input[0]
        highres_features = input[1]
        degradation_map = input[2]

        current_features = torch.cat([lowres_img, highres_features, degradation_map], 1)
        current_features = self.initial_convolution(current_features)
        for i in range(len(self.dense_blocks)):
            block_output = self.dense_blocks[i](current_features)
            current_features = current_features + block_output
        current_features = self.final_convolution(current_features)
        error_img = reconstruct_image(current_features, SCALE_FACTOR)
        return error_img













