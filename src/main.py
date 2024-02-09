import logging
import os

import data.loading as loading
import pandas as pd
import torch
from data.dataset import Forcasting_ERA5Dataset
from data.processing import select_data
from model.velocity import get_kernel, get_velocities
from torch.utils.data import DataLoader
from utils.loss import CustomGaussianNLLLoss

variables_time_dependant = ["t2m", "t", "z", "u10", "v10"]
variables_static = ["lsm", "orography"]

gpu_device = torch.device("cpu")  # fallback to cpu
if torch.cuda.is_available():
    gpu_device = torch.device("cuda")
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    gpu_device = torch.device("mps")
    torch.mps.empty_cache()

config = {
    "data_path_wb1": "data/era5_data/",
    "data_path_wb2": "data/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr",
    "freq": "6H",
    "periods": {
        "train": ("2006-01-01", "2015-12-31"),
        "val": ("2016-01-01", "2016-12-31"),
        "test": ("2017-01-01", "2018-12-31"),
    },
    "vel": {
        "rbf_alpha": 1.0,
        "stacking": 3,
        "bs": 50,
        "fitting_epoch": 200,
        "regul_coeff": 1e-7,
        "lr": 2,
        "device": gpu_device,
    },
    "model": {
        "emission_model": {
            "in_channels": 9 + 34,  # err_in
            "layers_length": [3, 2, 2],
            "layers_hidden_size": [
                128,
                64,
                2 * 9,
            ],  # 9 = out_types = len(paths_to_data)
        },
        "norm_type": "batch",
        "n_res_blocks": [3, 2, 2],
        "kernel_size": 3,
        "stride": 1,
        "dropout": 0.1,
    },
    "bs": 8,
    "device": gpu_device,
}

if __name__ == "__main__":
    # check the script is executed within the parent directory
    if not os.path.exists("src/main.py"):
        raise RuntimeError(
            "The script must be executed within the project root directory"
        )

    logging.basicConfig(level=logging.INFO)

    periods = {
        k: pd.date_range(*p, freq=config["freq"])
        for (k, p) in config["periods"].items()
    }
    raw_data = loading.wb1(config["data_path_wb1"], periods)
    # data = loading.wb2(config["data_path_wb2"], periods)

    logging.info("Raw data loaded, merged and normalized")
    logging.info("Raw data disk size: {} MiB".format(raw_data.nbytes / 1e6))

    data_selected = select_data(raw_data, periods)

    kernel = get_kernel(raw_data, config["vel"])
    data_velocities = get_velocities(data_selected, kernel, config)

    criterion = CustomGaussianNLLLoss()
    data = torch.cat([t.unsqueeze(-1) for t in data_selected["train"].values()], dim=-1)
    dataset = Forcasting_ERA5Dataset(data)
    train_loader = DataLoader(dataset, batch_size=config["bs"], shuffle=True)
