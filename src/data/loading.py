import logging

import numpy as np
import xarray as xr


def loading_wb1(data_path, periods):
    subfolders_list_path = [
        '2m_temperature',
        'temperature_850',
        'geopotential_500',
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
    ]

    constant_data_path = 'constants/constants_5.625deg.nc'

    # global_period period as datetime indexing xr.DataArray
    global_period = xr.DataArray(np.concatenate([*periods.values()], axis=0), dims='time')

    raw_data = []
    for subfolder in subfolders_list_path:
        dataset_path = data_path + subfolder + '/*.nc'
        data = xr.open_mfdataset(dataset_path, combine='by_coords')
        if 'level' in data.coords:
            data = data.drop_vars('level')
        raw_data.append(data)

    merged_raw_data = xr.merge(raw_data)
    merged_raw_data = merged_raw_data.sel(time=global_period)
    merged_raw_train_data = merged_raw_data.sel(time=periods['train'])

    # compute min and max
    min = merged_raw_train_data.min()
    max = merged_raw_train_data.max()

    # modify the data
    merged_raw_data = (merged_raw_data - min) / (max - min)

    # Add constants to the data
    constants = xr.open_dataset(data_path + constant_data_path)

    # keep only orography and land sea mask
    from main import variables_static
    constants = constants[variables_static]
    merged_raw_data = xr.merge([merged_raw_data, constants])

    # log info
    logging.info('Raw data loaded, merged and normalized')
    logging.info('Raw data disk size: {} MiB'.format(merged_raw_data.nbytes / 1e6))

    return merged_raw_data


def loading_wb2(data_path, periods):
    dict_var = {
        '2m_temperature': 't2m',
        'temperature': 't',
        'geopotential': 'z',
        '10m_u_component_of_wind': 'u10',
        '10m_v_component_of_wind': 'v10',
        'land_sea_mask': 'lsm',
        'geopotential_at_surface': 'orography',
    }

    global_period = xr.DataArray(np.concatenate([*periods.values()], axis=0), dims='time')
    raw_data = xr.open_zarr(data_path).sel(time=global_period)

    # extract data variables from dict_var keys and rename
    raw_data = raw_data[list(dict_var.keys())].rename(dict_var)
    raw_data = raw_data.rename({'latitude': 'lat', 'longitude': 'lon'})

    # keep only the right level
    raw_data['t'] = raw_data['t'].sel(level=850)
    raw_data['z'] = raw_data['z'].sel(level=500)
    raw_data = raw_data.drop_vars('level')

    from main import variables_time_dependant
    raw_data_variables = raw_data[variables_time_dependant]

    # compute min and max
    raw_data_train = raw_data_variables.sel(time=periods['train'])
    min = raw_data_train.min()
    max = raw_data_train.max()
    raw_data[variables_time_dependant] = (raw_data[variables_time_dependant] - min) / (max - min)

    # log info
    logging.info('Raw data loaded, merged and normalized')
    logging.info('Raw data disk size: {} MiB'.format(raw_data.nbytes / 1e6))

    return raw_data
