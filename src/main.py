import os
import argparse
import torch

from manager import Manager
from globals import SCALE_FACTOR

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Get command-line arguments.")
    arg_parser.add_argument('--mode', type=str, choices=['test', 'train'], default='test')

    main_path = os.path.dirname(os.path.abspath(__file__))

    # ------------------------------------------------
    # Testing arguments
    # ------------------------------------------------

    # Input folder for test images
    arg_parser.add_argument('--input', type=str, default=os.path.join(main_path, '..', 'input'))

    # Output folder for test results
    arg_parser.add_argument('--output', type=str, default=os.path.join(main_path, '..', 'output'))

    # Patch size for patch-based testing of large images.
    # Make sure the patch size is small enough that your GPU memory is sufficient.
    arg_parser.add_argument('--patch_size', type=int, default=200)

    # Checkpoint folder that contains the generator.pth and discriminator.pth checkpoint files.
    arg_parser.add_argument('--checkpoint_folder', type=str,
                            default=os.path.join(main_path, '..', 'checkpoints', 'x' + str(SCALE_FACTOR) + '_blind_sr'))

    # ------------------------------------------------
    # Training arguments
    # ------------------------------------------------

    # Log folder where Tensorboard logs are saved
    arg_parser.add_argument('--log_folder', type=str,
                            default=os.path.join(main_path, '..', 'logs', 'x' + str(SCALE_FACTOR) + '_blind_sr'))

    # Folders for training and validation datasets.
    arg_parser.add_argument('--train_input', type=str, default=os.path.join(main_path, '..', 'data', 'train'))
    arg_parser.add_argument('--valid_input', type=str, default=os.path.join(main_path, '..', 'data', 'validation'))

    # Define whether we use only the generator or the whole pipeline with the discriminator for training.
    arg_parser.add_argument('--network_type', type=str, choices=['generator', 'discriminator'], default='discriminator')

    arg_list = arg_parser.parse_args()
    manager = Manager(arg_list)
