import logging

import xarray as xr
import numpy as np


def loading_process(data_path, subfolders_list_path, constant_data_path,
                    train_period, val_period, test_period):
    # global_period period as datetime indexing xr.DataArray
    global_period = xr.DataArray(np.concatenate([train_period, val_period, test_period], axis=0), dims='time')

    raw_data = []
    for subfolder in subfolders_list_path:
        dataset_path = data_path + subfolder + '/*.nc'
        data = xr.open_mfdataset(dataset_path, combine='by_coords')
        if 'level' in data.coords:
            data = data.drop_vars('level')
        raw_data.append(data)

    merged_raw_data = xr.merge(raw_data)
    merged_raw_data = merged_raw_data.sel(time=global_period)
    merged_raw_train_data = merged_raw_data.sel(time=train_period)

    # compute min and max
    min = merged_raw_train_data.min()
    max = merged_raw_train_data.max()

    # modify the data
    merged_raw_data = (merged_raw_data - min) / (max - min)

    # Add constants to the data
    constants = xr.open_dataset(data_path + constant_data_path)
    # keep only orography and land sea mask
    constants = constants[['lsm', 'orography']]
    merged_raw_data = xr.merge([merged_raw_data, constants])

    # log info
    logging.info('Raw data loaded, merged and normalized')
    logging.info('Raw data disk size: {} MiB'.format(merged_raw_data.nbytes / 1e6))

    return merged_raw_data
