import numpy as np
import torch as th
import json
import torch
import datetime
import copy
import random

class MinMaxNormalization(object):
    """
        MinMax Normalization --> [-1, 1]
        x = (x - min) / (max - min).
        x = x * 2 - 1
    """

    def __init__(self):
        pass

    # Calculate min and max values from data
    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        # Normalize to [0,1] then to [-1,1]
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)  # First calculate min and max
        return self.transform(X) # Then normalize the data

    def inverse_transform(self, X):
        # Convert back from [-1,1] to original scale
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X


def data_load_single(args, dataset): 
    """
    args: Configuration object containing:
    task: Type of task (e.g., 'short')
    mode: Training mode (e.g., 'few-shot')
    few_ratio: Ratio for few-shot learning
    batch_size_1: Batch size for small data (default: 64)
    batch_size_2: Batch size for medium data (default: 32)
    batch_size_3: Batch size for large data (default: 16)
    """
    # Load data from JSON file
    folder_path = '../dataset/{}_{}.json'.format(dataset,args.task)
    f = open(folder_path,'r')
    data_all = json.load(f)

    # Load main data tensors
    """
        Converts JSON data to PyTorch tensors
        Adds channel dimension with unsqueeze(1)
        Shape: [N, 1, T, H, W] where:
        N: Number of samples
        1: Channel dimension
        T: Time steps
        H, W: Height and width
    """
    X_train = torch.tensor(data_all['X_train'][0])
    X_test = torch.tensor(data_all['X_test'][0])
    X_val = torch.tensor(data_all['X_val'][0])
    if X_train.ndim == X_test.ndim == X_val.ndim == 4:
        X_train = X_train.unsqueeze(1)
        X_test = X_test.unsqueeze(1)
        X_val = X_val.unsqueeze(1)
    elif X_train.ndim == X_test.ndim == X_val.ndim == 5:
        X_train = X_train.permute(0, 2, 1, 3, 4)
        X_test = X_test.permute(0, 2, 1, 3, 4)
        X_val = X_val.permute(0, 2, 1, 3, 4)
    else:
        raise ValueError(f"Input data should be 4D or 5D tensor, X_train {X_train.ndim}, X_test {X_test.ndim}, X_val {X_val.ndim}")
    print("data_load_single X_train shape:", X_train.shape)
    print("data_load_single X_test shape:", X_test.shape)
    print("data_load_single X_val shape:", X_val.shape)

    # Load periodic data and rearrange dimensions
    """
        Loads periodic patterns in data
        Rearranges dimensions using permute
        Shape: [N, C, T, H, W] after permute
        N: Batch size
        C: Channels - Number of features or measurements per spatial location
        T: Time steps
        H, W: Height and width (Spatial dimension - height/width of the grid)
    """
    X_train_period = torch.tensor(data_all['X_train'][1]).permute(0,2,1,3,4)
    X_test_period = torch.tensor(data_all['X_test'][1]).permute(0,2,1,3,4)
    X_val_period = torch.tensor(data_all['X_val'][1]).permute(0,2,1,3,4)

    #Stores sequence length and spatial dimensions
    #Used later for batch size selection
    args.seq_len = X_train.shape[2]
    H, W = X_train.shape[3], X_train.shape[4]  

    # Handles different timestamp formats for different datasets
    if 'TaxiBJ' in dataset:
        X_train_ts = data_all['timestamps']['train']
        X_test_ts = data_all['timestamps']['test']
        X_val_ts = data_all['timestamps']['val']

        # Convert to (weekday, hour) format
        X_train_ts = torch.tensor([[(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').weekday(),datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').hour*2+int(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').minute>=30)) for i in t] for t in X_train_ts])
        X_test_ts = torch.tensor([[(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').weekday(),datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').hour*2+int(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').minute>=30)) for i in t] for t in X_test_ts])
        X_val_ts = torch.tensor([[(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').weekday(),datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').hour*2+int(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').minute>=30)) for i in t] for t in X_val_ts])

    elif 'Crowd' in dataset or 'Cellular' in dataset or 'Traffic_log' in dataset:
        X_train_ts = data_all['timestamps']['train']
        X_test_ts = data_all['timestamps']['test']
        X_val_ts = data_all['timestamps']['val']

        # Convert to (weekday, time_slot) format
        X_train_ts = torch.tensor([[((i%(24*2*7)//(24*2)+2)%7,i%(24*2)) for i in t] for t in X_train_ts])
        X_test_ts = torch.tensor([[((i%(24*2*7)//(24*2)+2)%7, i%(24*2)) for i in t] for t in X_test_ts])
        X_val_ts = torch.tensor([[((i%(24*2*7)//(24*2)+2)%7, i%(24*2)) for i in t] for t in X_val_ts])

    elif 'TaxiNYC' in dataset or 'BikeNYC' in dataset or 'TDrive' in dataset or 'Traffic' in dataset or 'DC' in dataset or 'Austin' in dataset or 'Porto' in dataset or 'CHI' in dataset or 'METR-LA' in dataset or 'CrowdBJ' in dataset:
        # Direct conversion to tensor
        X_train_ts = torch.tensor(data_all['timestamps']['train'])
        X_test_ts = torch.tensor(data_all['timestamps']['test'])
        X_val_ts = torch.tensor(data_all['timestamps']['val'])

    elif 'MciTRT' in dataset:
        # Direct tensor conversion
        print("Loading custom dataset")
        X_train_ts = torch.tensor(data_all['timestamps']['train'])
        X_test_ts = torch.tensor(data_all['timestamps']['test'])
        X_val_ts = torch.tensor(data_all['timestamps']['val'])

    else:
        raise NotImplementedError(f"Not implemented dataset {dataset}")

    my_scaler = MinMaxNormalization()
    MAX = max(torch.max(X_train).item(), torch.max(X_test).item(), torch.max(X_val).item())
    MIN = min(torch.min(X_train).item(), torch.min(X_test).item(), torch.min(X_val).item())
    my_scaler.fit(np.array([MIN, MAX]))

    # Normalizes all data to [-1, 1] range
    # Reshapes for normalization then back to original shape
    """
        Why Reshape?
        - Normalization requires 2D data (features x samples)
        - Need to convert 5D data (N, C, T, H, W) to 2D
        - Reshape to (N*C*T, H*W) then normalize
        - Reshape back to original shape
    """
    X_train = my_scaler.transform(X_train.reshape(-1,1)).reshape(X_train.shape)
    X_test = my_scaler.transform(X_test.reshape(-1,1)).reshape(X_test.shape)
    X_val = my_scaler.transform(X_val.reshape(-1,1)).reshape(X_val.shape)
    X_train_period = my_scaler.transform(X_train_period.reshape(-1,1)).reshape(X_train_period.shape)
    X_test_period = my_scaler.transform(X_test_period.reshape(-1,1)).reshape(X_test_period.shape)
    X_val_period = my_scaler.transform(X_val_period.reshape(-1,1)).reshape(X_val_period.shape)

    """
        Creates list of data for each dataset
        Each list contains:
        - Original data (X_train[i], X_test[i], X_val[i])
        - Timestamps (X_train_ts[i], X_test_ts[i], X_val_ts[i])
        - Periodic patterns (X_train_period[i], X_test_period[i], X_val_period[i])
    """
    data = [[X_train[i], X_train_ts[i], X_train_period[i]] for i in range(X_train.shape[0])]
    test_data = [[X_test[i], X_test_ts[i], X_test_period[i]] for i in range(X_test.shape[0])]
    val_data = [[X_val[i], X_val_ts[i], X_val_period[i]] for i in range(X_val.shape[0])]


    if args.mode == 'few-shot':
        # Reduces training data size for few-shot learning
        data = data[:int(len(data)*args.few_ratio)]

    """
        Selects batch size based on spatial dimensions (Smaller batch size for larger data)
        batch_size_1: Batch size for small data (default: 64)
        batch_size_2: Batch size for medium data (default: 32)
        batch_size_3: Batch size for large data (default: 16)
    """
    if H + W < 32:
        batch_size = args.batch_size_1
    elif H + W < 48:
        batch_size = args.batch_size_2
    elif H + W < 64:
        batch_size = args.batch_size_3
    else:
        batch_size = args.batch_size_3

    """
        Creates DataLoader objects for training, testing, and validation data
        num_workers: Number of worker threads for loading data
        batch_size: Batch size for each loader
        shuffle: Whether to shuffle the data
    """
    data = th.utils.data.DataLoader(data, num_workers=4, batch_size=batch_size, shuffle=True) 
    test_data = th.utils.data.DataLoader(test_data, num_workers=4, batch_size = 4 * batch_size, shuffle=False)
    val_data = th.utils.data.DataLoader(val_data, num_workers=4, batch_size = 4 * batch_size, shuffle=False)

    return  data, test_data, val_data, my_scaler

"""
    Input:
        args: Configuration object containing:
        dataset: String of dataset names separated by '*' (e.g., "TaxiBJCrowdNYC")
    Output:
        data_all: List of datasets with their corresponding DataLoaders and scalers
        test_data_all: List of test datasets with their corresponding DataLoaders and scalers
        val_data_all: List of validation datasets with their corresponding DataLoaders and scalers
        my_scaler_all: Dictionary of scalers for each dataset
"""
def data_load(args):

    data_all = []
    test_data_all = []
    val_data_all = []
    my_scaler_all = []
    my_scaler_all = {}

    for dataset_name in args.dataset.split('*'):
        data, test_data, val_data, my_scaler = data_load_single(args,dataset_name)
        data_all.append([dataset_name, data])
        test_data_all.append(test_data)
        val_data_all.append(val_data)
        my_scaler_all[dataset_name] = my_scaler

    data_all = [(name,i) for name, data in data_all for i in data]
    
    # Shuffle combined data
    random.seed(1111)
    random.shuffle(data_all)
    
    return data_all, test_data_all, val_data_all, my_scaler_all


def data_load_main(args):

    data, test_data, val_data, scaler = data_load(args)

    return data, test_data, val_data, scaler

