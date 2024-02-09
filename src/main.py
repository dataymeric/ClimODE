import logging
import os

import pandas as pd
import torch
from model.models import ClimODE
from model.velocity import get_kernel, get_velocities
from torch import optim
from torch.utils.data import DataLoader
from utils.loss import CustomGaussianNLLLoss

import data.loading as loading
from data.dataset import Forcasting_ERA5Dataset
from data.embeddings import get_time_localisation_embeddings
from data.processing import select_data

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
    "freq": 6,  # In hours
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
        "VelocityModel": {
            "local": {
                "in_channels": 30 + 34,  # 34 d'embeding, 30 = jsp
                "layers_length": [5, 3, 2],
                "layers_hidden_size": [128, 64, 2 * 5],
                # 5 = out_types = len(paths_to_data)
            },
            "global": {
                "in_channels": 30 + 34,
                "out_channels": 2 * 5,
            },
            "gamma": 0.1,
        },
        "EmissionModel": {
            "in_channels": 5 + 34,  # err_in ; 5 ou 9 ??? je sais plus faut vérif
            "layers_length": [3, 2, 2],
            "layers_hidden_size": [
                128,
                64,
                2 * 5,
            ],  # 5 = out_types = len(paths_to_data)
        },
        "norm_type": "batch",
        "n_res_blocks": [3, 2, 2],
        "kernel_size": 3,
        "stride": 1,
        "dropout": 0.1,
    },
    "bs": 8,
    "max_epoch": 300,
    "lr": 0.0005,
    "device": gpu_device,
}

if __name__ == "__main__":
    # check the script is executed within the parent directory

    logging.basicConfig(level=logging.INFO)

    periods = {
        k: pd.date_range(*p, freq=str(config["freq"]) + "H")
        for (k, p) in config["periods"].items()
    }
    raw_data = loading.wb1(config["data_path_wb1"], periods)
    train_raw_data = raw_data.sel(time=periods["train"])
    # data = loading.wb2(config["data_path_wb2"], periods)

    logging.info("Raw data loaded, merged and normalized")
    logging.info("Raw data disk size: {} MiB".format(raw_data.nbytes / 1e6))

    data_selected = select_data(raw_data, periods)

    kernel = get_kernel(raw_data, config["vel"])
    data_velocities = get_velocities(data_selected, kernel, config)
    train_velocities = torch.cat(tuple(data_velocities["train"].values()), dim=1).view(
        -1, 32, 64, 10
    )  # (1826, 10, 32, 64) -> (1826, 32, 64, 10) pour compatibilité avec les futurs cat

    train_data = torch.cat(
        [t.unsqueeze(-1) for t in data_selected["train"].values()], dim=-1
    )
    dataset = Forcasting_ERA5Dataset(train_data)
    train_loader = DataLoader(dataset, batch_size=config["bs"], shuffle=True)

    time_step = torch.Tensor(list(range(len(train_data))))
    time_step = torch.arange(0, len(train_data), 1)
    time_step = torch.Tensor([0, 1])
    time_pos_embedding = get_time_localisation_embeddings(
        time_step,
        torch.tensor(train_raw_data["lat"].values),
        torch.tensor(train_raw_data["lon"].values),
        torch.tensor(train_raw_data["lsm"].values),
        torch.tensor(train_raw_data["orography"].values),
    ).float()  # float64 to float32 (important for conv) TODO
    model = ClimODE(config, time_pos_embedding)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    criterion = CustomGaussianNLLLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300)
