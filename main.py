import os

# stop tensorboard warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# stop W&B logs
os.environ['WANDB_SILENT'] = 'true'

import argparse
from train import train
from eval import eval
from auto_aug import search
from utils.config_parser import get_config_data
from utils.check_gpu import get_training_device

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("experiment_file",
                    help="The name of the experiment config file")
parser.add_argument('-p', '--publish', action='store_true', 
    help="publishes results to telegram")
parser.add_argument('-a', '--auto_aug', action='store_true', 
    help="Search for augmentations using auto augmentation")
args = parser.parse_args()

# Get experiment config values
if args.experiment_file is None:
    exit()
config = get_config_data(args.experiment_file, args.publish)

# Get GPU / CPU device instance
device = get_training_device()

if config['mode'] == 'test':
    eval( config, device )
elif config['mode'] == 'train':
    if args.auto_aug:
        search( config, device )
    else:
        train( config, device )
else:
    print("[ Experiment Mode should either be train/test ]")
    exit()
