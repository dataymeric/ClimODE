import logging
import os

import numpy as np
import pandas as pd
import torch.utils.data
from tensordict import TensorDict

from model.velocity import get_gauss_kernel, stack_past_samples, select_from_slice_for_batch, get_batch_velocity
from data.loading import loading_process

data_path = '../ClimODE_original/era5_data/'

variables_time_dependant = ['t2m', 't', 'z', 'u10', 'v10']
variables_static = ['lsm', 'orography']

subfolders_list = [
    '2m_temperature',
    'temperature_850',
    'geopotential_500',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
]

constant_data_path = 'constants/constants_5.625deg.nc'

sample_freq = '6H'
train_period = pd.date_range(start='2006-01-01', end='2015-12-31', freq=sample_freq)
val_period = pd.date_range(start='2016-01-01', end='2016-12-31', freq=sample_freq)
test_period = pd.date_range(start='2017-01-01', end='2018-12-31', freq=sample_freq)

stack_velocity_depth = 3
bs = 8
bs_vel = 17

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    torch.mps.empty_cache()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    data = loading_process(data_path, subfolders_list, constant_data_path, train_period, val_period, test_period)

    lat = data.coords.variables['lat'].values.astype(np.float32)
    lon = data.coords.variables['lon'].values.astype(np.float32)

    kernel = get_gauss_kernel(lat, lon, alpha=1.0)
    kernel = kernel.to(device).expand(bs_vel, *kernel.shape)

    data_train = data.sel(time=train_period)[variables_time_dependant]
    data_train = TensorDict(
        source={k: data_train[k].values for k in data_train.data_vars},
        batch_size=data_train.sizes['time']
    )

    new_batch_size = data_train.batch_size[0] - stack_velocity_depth + 1
    data_train = data_train.apply(lambda x: stack_past_samples(x, n=stack_velocity_depth), batch_size=[new_batch_size])

    init_data_train_batch_size = data_train.batch_size[0] // bs + (data_train.batch_size[0] % bs > 0)
    init_data_train = data_train.apply(lambda x: select_from_slice_for_batch(x, batch_size=bs),
                                       batch_size=[init_data_train_batch_size])

    dataloader = torch.utils.data.DataLoader(init_data_train, batch_size=bs_vel, shuffle=False, collate_fn=lambda x: x, pin_memory=True)

    velocities: TensorDict = torch.cat([get_batch_velocity(batch, kernel, device) for batch in dataloader], dim=0)

    velocities_save_path = f'velocities/velocities_bs{bs}'
    os.makedirs(velocities_save_path, exist_ok=True)

    for k in velocities.sorted_keys:
        torch.save(velocities[k], velocities_save_path+f'/velocity_{k}.pt')
