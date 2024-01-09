import logging
import os

import pandas as pd
import torch.utils.data

from data.loading import loading_wb1
from data.processing import select_data
from model.velocity import get_velocities, get_kernel

variables_time_dependant = ['t2m', 't', 'z', 'u10', 'v10']
variables_static = ['lsm', 'orography']

gpu_device = torch.device('cpu')  # fallback to cpu
if torch.cuda.is_available():
    gpu_device = torch.device('cuda')
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    gpu_device = torch.device('mps')
    torch.mps.empty_cache()

config = {
    'data_path_wb1': 'data/era5_data/',
    'data_path_wb2': 'data/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr',
    'freq': '6H',
    'periods': {
        'train': ('2006-01-01', '2015-12-31'),
        'val': ('2016-01-01', '2016-12-31'),
        'test': ('2017-01-01', '2018-12-31')
    },
    'vel': {
        'rbf_alpha': 1.0,
        'stacking': 3,
        'bs': 50,
        'fitting_epoch': 200,
        'regul_coeff': 1e-7,
        'lr': 2,
        'device': gpu_device,
    },
    'bs': 8,
    'device': gpu_device
}

if __name__ == '__main__':

    # check the script is executed within the parent directory
    if not os.path.exists('src/main.py'):
        raise RuntimeError('The script must be executed within the project root directory')

    logging.basicConfig(level=logging.INFO)

    periods = {k: pd.date_range(*p, freq=config['freq']) for (k, p) in config['periods'].items()}
    data = loading_wb1(config['data_path_wb1'], periods)
    # data = loading_wb2(config['data_path_wb2'], periods)

    data_selected = select_data(data, periods)

    kernel = get_kernel(data, config['vel'])
    data_velocities = get_velocities(data_selected, kernel, config)
