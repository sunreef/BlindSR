import os
import torch
import tensorboardX
from torch.utils.data import DataLoader

from model import Generator
from dataset import TrainDataset

class Manager:
    def __init__(self, args):
        self.args = args
        print(self.args)

        self.init_model()
        self.init_train_data()

    def init_model(self):
        self.generator = Generator()
        self.generator.cuda()


    def init_train_data(self):
        train_folder = self.args.train_input
        train_dataset = TrainDataset(train_folder)

        train_dataloader = DataLoader(train_dataset, batch_size = 1)

        for batch in train_dataloader:
            print("batch")












