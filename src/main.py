import logging
import os
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any, Optional

import fire
from ignite.contrib import handlers
from ignite.contrib.handlers.wandb_logger import OutputHandler
from ignite.utils import manual_seed, setup_logger

from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver
from ignite.contrib.handlers import global_step_from_engine, WandBLogger
import ignite.distributed as idist

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
    gpu_device = torch.device("cpu")
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
    "patience": 10,
    "device": gpu_device,
    "n_saved": 100,
    "seed": 42,
    "save_every_iter": 50,
    "output_dir": "output",
    "filename_prefix": "saved_checkpoint",
    "log_every_iter": 5,
}


def setup_handlers(
    trainer: Engine,
    evaluator: Engine,
    config: Any,
    to_save_train: Optional[dict] = None,
    to_save_eval: Optional[dict] = None,
):
    """Setup Ignite handlers."""

    ckpt_handler_train = ckpt_handler_eval = None
    # checkpointing
    saver = DiskSaver(config["output_dir"] / "checkpoints", require_empty=False)
    ckpt_handler_train = Checkpoint(
        to_save_train,
        saver,
        filename_prefix=config["filename_prefix"],
        n_saved=config["n_saved"],
    )

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=config["save_every_iter"]),
        ckpt_handler_train,
    )
    global_step_transform = None

    if to_save_train.get("trainer", None) is not None:
        global_step_transform = global_step_from_engine(to_save_train["trainer"])

    ckpt_handler_eval = Checkpoint(
        to_save_eval,
        saver,
        filename_prefix="best",
        n_saved=config["n_saved"],
        global_step_transform=global_step_transform,
        score_name="eval_accuracy",
        score_function=lambda engine: -1.0 * engine.state.output,
    )
    evaluator.add_event_handler(Events.EPOCH_COMPLETED(every=1), ckpt_handler_eval)

    return ckpt_handler_train, ckpt_handler_eval


def setup_output_dir(config: Any, rank: int) -> Path:
    """Create output folder."""
    output_dir = config["output_dir"]
    if rank == 0:
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{now}-lr-{config['lr']}"
        path = Path(config["output_dir"], name)
        path.mkdir(parents=True, exist_ok=True)
        output_dir = path.as_posix()
    return Path(idist.broadcast(output_dir, src=0))


def setup_exp_logging(config, trainer, evaluator, optimizer):
    """Setup Experiment Tracking logger from Ignite."""

    wandb_logger = WandBLogger(
        project="ClimODE",
        config=config,
    )

    wandb_logger.attach_opt_params_handler(
        trainer, event_name=Events.ITERATION_STARTED, optimizer=optimizer
    )

    wandb_logger.attach(
        evaluator,
        log_handler=OutputHandler(
            tag="validation",
            output_transform=lambda x: {"loss": x},
            global_step_transform=lambda x, _: trainer.state.iteration,
        ),
        event_name=Events.EPOCH_COMPLETED,
    )

    wandb_logger.attach(
        trainer,
        log_handler=OutputHandler(
            tag="training",
            output_transform=lambda x: {"loss": x},
            global_step_transform=lambda x, _: trainer.state.iteration,
        ),
        event_name=Events.ITERATION_COMPLETED,
    )

    return wandb_logger


def setup_logging(config: Any) -> logging.Logger:
    """Setup logger with `ignite.utils.setup_logger()`.

    Parameters
    ----------
    config
        config object. config has to contain `verbose` and `output_dir` attribute.

    Returns
    -------
    logger
        an instance of `Logger`
    """
    green = "\033[32m"
    reset = "\033[0m"
    logger = setup_logger(
        name=f"{green}[ignite]{reset}",
        level=logging.DEBUG,
        filepath=config["output_dir"] / "training-info.log",
    )
    return logger


def run(local_rank: int, config: Any):
    # make a certain seed
    rank = idist.get_rank()
    manual_seed(config["seed"] + rank)

    config["output_dir"] = setup_output_dir(config, rank)

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
        data_selected["train"][config["vel"]["stacking"] - 1 :],
        data_velocities["train"],
        pred_length=config["pred_length"],
    )

    val_dataset = Forcasting_ERA5Dataset(
        data_selected["val"][config["vel"]["stacking"] - 1 :],
        data_velocities["val"],
        pred_length=config["pred_length"],
    )

    test_dataset = Forcasting_ERA5Dataset(
        data_selected["test"][config["vel"]["stacking"] - 1 :],
        data_velocities["test"],
        pred_length=config["pred_length"],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["bs"], shuffle=True, collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config["bs"], shuffle=True, collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config["bs"], shuffle=True, collate_fn=collate_fn
    )

    leap_year = train_raw_data["time"].to_dataframe().index.is_leap_year
    day_of_year_ratio = train_raw_data["time"].dt.dayofyear / (365 + leap_year)
    hour_of_day = train_raw_data["time"].dt.hour

    device = idist.device()

    time_pos_embedding = get_time_localisation_embeddings(
        torch.tensor(day_of_year_ratio.values),
        torch.tensor(hour_of_day.values),
        torch.tensor(train_raw_data["lat"].values),
        torch.tensor(train_raw_data["lon"].values),
        torch.tensor(train_raw_data["lsm"].values),
        torch.tensor(train_raw_data["orography"].values),
    ).float()  # .to(device=device)  # if enough VRAM

    model = ClimODE(config, time_pos_embedding).to(device)
    model = idist.auto_model(model)

    optimizer = optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    optimizer = idist.auto_optim(optimizer)

    criterion = CustomGaussianNLLLoss().to(device=device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300)

    # print total number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def eval_step(engine, batch):
        epoch = engine.state.epoch
        if epoch == 0:
            var_coeff = 0.001
        else:
            var_coeff = 2 * scheduler.get_last_lr()[0]
        model.eval()
        with torch.no_grad():
            (data, vel, t) = batch
            data = data.to(device)
            vel = vel.to(device)
            mean, std = model(data, vel, t)
            loss = criterion(mean, data, std, var_coeff)
            return loss.item()

    def train_step(engine, batch):
        epoch = engine.state.epoch
        if epoch == 0:
            var_coeff = 0.001
        else:
            var_coeff = 2 * scheduler.get_last_lr()[0]

        model.train()
        optimizer.zero_grad()
        (data, vel, t) = batch
        data = data.to(device)
        vel = vel.to(device)
        mean, std = model(data, vel, t)
        loss = criterion(mean, data, std, var_coeff)
        loss.backward()
        optimizer.step()
        return loss.item()

    trainer = Engine(train_step)
    evaluator = Engine(eval_step)

    logger = setup_logging(config)
    logger.info("Configuration: \n%s", pformat(config))
    trainer.logger = evaluator.logger = logger

    logger.info(f"{total_params:,} trainable parameters.")

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_lr(engine):
        scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_loss(engine):
        evaluator.run(val_loader, max_epochs=1)

    to_save_eval = {"model": model, "optimizer": optimizer, "scheduler": scheduler}

    to_save_train = {"model": model}
    ckpt_handler_train, ckpt_handler_eval = setup_handlers(
        trainer, evaluator, config, to_save_train, to_save_eval
    )

    # experiment tracking
    if rank == 0:
        exp_logger = setup_exp_logging(config, trainer, evaluator, optimizer)

    trainer.run(train_loader, max_epochs=config["max_epoch"])

    # close logger
    if rank == 0:
        exp_logger.close()

    # show last checkpoint names
    logger.info(
        "Last training checkpoint name - %s",
        ckpt_handler_train.last_checkpoint,
    )

    logger.info(
        "Last evaluation checkpoint name - %s",
        ckpt_handler_eval.last_checkpoint,
    )


def main(backend=None, **spawn_kwargs):
    with idist.Parallel(backend=backend, **spawn_kwargs) as parallel:
        parallel.run(run, config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire({"run": main})
