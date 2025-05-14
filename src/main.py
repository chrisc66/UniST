import argparse
import random
import os
from model import UniST_model
from train import TrainLoop

import setproctitle
import torch

from DataLoader import data_load_main
from utils import *

import torch as th
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def setup_init(seed):
    random.seed(seed) # Set Python's random seed
    os.environ['PYTHONHASHSEED'] = str(seed) # Set Python's hash seed
    np.random.seed(seed) # Set NumPy's random seed
    th.manual_seed(seed) # Set PyTorch's manual seed
    th.cuda.manual_seed(seed) # Set PyTorch's CUDA seed
    th.backends.cudnn.benchmark = False # Disable cuDNN benchmarking
    th.backends.cudnn.deterministic = True # Enable cuDNN deterministic mode

# Get the device to use for torch.distributed.
def dev(device_id='0'):
    """
    Get the device to use for torch.distributed.
    # """
    if th.cuda.is_available():  # Check if GPU is available
        return th.device('cuda:{}'.format(device_id)) # Return CUDA device if available
    return th.device("cpu") # Return CPU device if GPU is not available 

# Defines default configuration parameters
def create_argparser():
    defaults = dict(
        # experimental settings
        task = 'short', # ['short','long']
        dataset = 'Crowd', # ['TaxiBJ','TaxiBJCrowdNYC','TaxiBJ_CrowdNYC','TaxiBJ_CrowdBJ','TaxiBJ_CrowdBJ_NYC']
        mode='training', # ['training','prompting','testing']
        file_load_path = '',
        used_data = '', # ['train','test','val']
        process_name = 'process_name', # Name of the process
        prompt_ST = 0, # Whether to use prompt-tuning
        his_len = 6, # Number of historical time steps
        pred_len = 6, # Number of prediction time steps
        few_ratio = 0.5, # Ratio of few-shot learning
        stage = 0, # Stage of training

        # model settings
        mask_ratio = 0.5, # Ratio of masked time steps
        patch_size = 2, # Patch size
        t_patch_size = 2, # Patch size for temporal dimension
        size = 'middle', # ['small','middle','large']
        no_qkv_bias = 0, # Whether to use bias in QKV transformation
        pos_emb = 'SinCos', # ['SinCos','Learned']
        num_memory_spatial = 512, # Number of spatial memory units
        num_memory_temporal = 512, # Number of temporal memory units
        conv_num = 3, # Number of convolutional layers
        prompt_content = 's_p_c', # Prompt content type: ['s_p_c','s_p_c_t']

        # pretrain settings
        random=True, # Whether to use random masking
        mask_strategy = 'random', # ['random','causal','frame','tube']
        mask_strategy_random = 'batch', # ['none','batch']
        
        # training parameters
        lr=1e-3, # Learning rate
        min_lr = 1e-5, # Minimum learning rate
        early_stop = 5, # Early stopping patience
        weight_decay=1e-6, # Weight decay
        batch_size=256, # Batch size
        log_interval=20, # Log interval
        total_epoches = 200, # Total number of epochs
        device_id='0', # Device ID
        machine = 'machine_name', # Machine name
        clip_grad = 0.05, # Gradient clipping threshold
        lr_anneal_steps = 200, # Learning rate annealing steps
        batch_size_1 = 64, # Batch size for first stage
        batch_size_2 = 32, # Batch size for second stage
        batch_size_3 = 16, # Batch size for third stage
        loss_channel_weights = [1.0, 1.0], # Weights for each channel in loss calculation
    )
    parser = argparse.ArgumentParser() # Create argument parser
    add_dict_to_argparser(parser, defaults) # Add default parameters to parser
    return parser
    
# PyTorch handles sharing data between processes
torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    # Enable anomaly detection for PyTorch
    th.autograd.set_detect_anomaly(True)
    
    args = create_argparser().parse_args() # Parse command-line arguments
    setproctitle.setproctitle("{}-{}".format(args.process_name, args.device_id)) # Set process title
    setup_init(100) # Initialize random seeds

    data, test_data, val_data, args.scaler = data_load_main(args) # Load data
    assert args.his_len + args.pred_len == args.seq_len # Check sequence length

     # Set mode based on few-shot ratio
    if args.few_ratio < 1.0: # Check if few-shot learning is used
        if args.few_ratio == 0.0:
            args.mode = 'testing' # just evaluation on the test set
        else:
            args.mode = 'prompting' # with prompt-tuning

    # Set folder name based on mode and few-shot ratio
    args.folder = 'Dataset_{}_Task_{}_FewRatio_{}/'.format(args.dataset, args.task, args.few_ratio)

    # Set folder name based on mode and few-shot ratio
    if args.mode in ['training','prompting']:
        if args.prompt_ST != 0:
            args.folder = 'Prompt_'+args.folder
        else:
            args.folder = 'Pretrain_'+args.folder
    else:
        args.folder = 'Test_'+args.folder

    # Set model path and create directories
    args.model_path = './experiments/{}'.format(args.folder) 
    logdir = "./logs/{}".format(args.folder)
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
        os.mkdir(args.model_path+'model_save/')

    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir = logdir,flush_secs=5)
    device = dev(args.device_id)

    model = UniST_model(args=args).to(device) # Initialize model

    # Handle prompt spatio-temporal mode
    if args.prompt_ST==1:
        if args.file_load_path != '':
            # Load pretrained model
            model.load_state_dict(torch.load('{}.pkl'.format(args.file_load_path),map_location=device), strict=False)
            print('pretrained model loaded') 
        # Set mask strategy for prompt spatio-temporal mode
        args.mask_strategy_random = 'none'
        args.mask_strategy = 'temporal'
        args.mask_ratio = (args.pred_len+0.0) / (args.pred_len+args.his_len)

    # Start training loop
    TrainLoop(
        args = args, # Arguments
        writer = writer, # Tensorboard writer
        model=model, # Model
        data=data, # Training data
        test_data=test_data, # Test data
        val_data=val_data, # Validation data
        device=device, # Device
        early_stop = args.early_stop, # Early stopping patience
    ).run_loop()

if __name__ == "__main__":
    main()