import os
import torch
import tensorboardX

from model import Generator

class Manager:
    def __init__(self, args):
        self.args = args
        self.generator = Generator()
        self.generator.cuda()
        print(self.args)






