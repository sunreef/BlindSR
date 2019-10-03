import torch

from globals import KERNEL_SIZE, SCALE_FACTOR
from dense_block import DenseBlock

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # self.initial_convolution = torch.nn.Conv2d(6 + KERNEL_SIZE * KERNEL_SIZE, 64, 3, padding=1)
        self.initial_convolution = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU()
        )

        self.dense_blocks = [
            DenseBlock(64, 16, 5) for i in range(2 * SCALE_FACTOR)
        ]

        self.final_convolution = torch.nn.Sequential(
            torch.nn.Conv2d(64, 3 * SCALE_FACTOR * SCALE_FACTOR, 3, padding=1),
            torch.nn.ReLU()
        )

    def forward(self, *input):
        lowres_img = input[0]
        current_features = lowres_img
        current_features = self.initial_convolution(current_features)
        for i in range(len(self.dense_blocks)):
            block_output = self.dense_blocks[i](current_features)
            current_features = current_features + block_output
        return current_features







