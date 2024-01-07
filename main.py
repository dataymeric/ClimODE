import logging

import pandas as pd

from data.loading import loading_process

data_path = '../ClimODE_original/era5_data/'

subfolders_list = [
    '2m_temperature',
    'temperature_850',
    'geopotential_500',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
]

constant_data_path = 'constants/constants_5.625deg.nc'

train_period = pd.date_range(start='2006-01-01', end='2015-12-31', freq='6H')
val_period = pd.date_range(start='2016-01-01', end='2016-12-31', freq='6H')
test_period = pd.date_range(start='2017-01-01', end='2018-12-31', freq='6H')

logging.basicConfig(level=logging.INFO)
data = loading_process(data_path, subfolders_list, constant_data_path, train_period, val_period, test_period)

