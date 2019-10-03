import os
import argparse
import torch

from manager import Manager

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Get command-line arguments for input and output folders if there are any.")
    arg_parser.add_argument('--mode', type=str, choices=['test', 'train'], default='test')

    main_path = os.path.dirname(os.path.abspath(__file__))
    arg_parser.add_argument('--input', type=str,default=os.path.join(main_path, '..', 'input'))
    arg_parser.add_argument('--output', type=str,default=os.path.join(main_path, '..', 'output'))

    arg_list = arg_parser.parse_args()
    manager = Manager(arg_list)

