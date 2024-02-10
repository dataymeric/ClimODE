import logging
import os

from icecream import ic, install
from torch import optim

install()
ic.configureOutput(includeContext=True)

import pandas as pd
import torch
from model.models import ClimODE
from model.velocity import get_kernel, get_velocities
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.loss import CustomGaussianNLLLoss

import data.loading as loading
from data.dataset import Forcasting_ERA5Dataset, collate_fn
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
    "freq": 6,
    "nb_variable_time_dependant": len(variables_time_dependant),
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
            "in_channels": 9 + 34,  # err_in ; je sais pas pourquoi 9
            "layers_length": [3, 2, 2],
            "layers_hidden_size": [
                128,
                64,
                2 * len(variables_time_dependant),
            ],  # 5 = out_types = len(paths_to_data)
        },
        "norm_type": "batch",
        "n_res_blocks": [3, 2, 2],
        "kernel_size": 3,
        "stride": 1,
        "dropout": 0.1,
    },
    "pred_length": 8,
    "weight_decay": 1e-5,
    "bs": 12,
    "max_epoch": 300,
    "lr": 0.0005,
    "device": gpu_device,
}

if __name__ == "__main__":
    # check the script is executed within the parent directory

    logging.basicConfig(level=logging.INFO)

    periods = {
        k: pd.date_range(*p, freq=str(config["freq"]) + "h")
        for (k, p) in config["periods"].items()
    }
    raw_data = loading.wb1(config["data_path_wb1"], periods)
    # data = loading.wb2(config["data_path_wb2"], periods)
    train_raw_data = raw_data.sel(time=periods["train"])

    logging.info("Raw data loaded, merged and normalized")
    logging.info("Raw data disk size: {} MiB".format(raw_data.nbytes / 1e6))

    data_selected = select_data(raw_data, periods)

    kernel = get_kernel(raw_data, config["vel"])
    data_velocities = get_velocities(data_selected, kernel, config)

    train_dataset = Forcasting_ERA5Dataset(
        data_selected["train"],
        data_velocities["train"],
        pred_length=config["pred_length"],
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config["bs"], shuffle=True, collate_fn=collate_fn
    )

    leap_year = train_raw_data["time"].to_dataframe().index.is_leap_year
    day_of_year_ratio = train_raw_data["time"].dt.dayofyear / (365 + leap_year)
    hour_of_day = train_raw_data["time"].dt.hour

    time_pos_embedding = get_time_localisation_embeddings(
        torch.tensor(day_of_year_ratio.values),
        torch.tensor(hour_of_day.values),
        torch.tensor(train_raw_data["lat"].values),
        torch.tensor(train_raw_data["lon"].values),
        torch.tensor(train_raw_data["lsm"].values),
        torch.tensor(train_raw_data["orography"].values),
    ).float()

    ic(time_pos_embedding.shape)
    criterion = CustomGaussianNLLLoss()

    model = ClimODE(config, time_pos_embedding).to(config["device"])
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = CustomGaussianNLLLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300)

    print("device", gpu_device)

    for epoch in range(config["max_epoch"]):
        print(f"############ EPOCH {epoch}")
        if epoch == 0:
            var_coeff = 0.001
        else:
            var_coeff = 2 * scheduler.get_last_lr()[0]

        for data, vel, t in tqdm(train_loader):
            optimizer.zero_grad()
            # data : torch.Size([12, 8, 5, 32, 64])
            # vel : torch.Size([12, 5, 2, 32, 64])
            # t : torch.Size([12]) # index à utiliser pour les embeddings
            data = data.to(config["device"])
            vel = vel.to(config["device"])

            mean, std = model(data[:, 0], vel, t)

            loss = criterion(mean, data, std, var_coeff)
            if loss.isnan().any():
                raise ValueError("Loss have NaN")
            loss.backward()
            optimizer.step()

            print("loss:", loss.item())
        scheduler.step()
