import logging
import os

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from tqdm import tqdm


def get_gauss_kernel(lat, lon, alpha):
    """
    Creates and inverts a Gaussian RBF kernel matrix based on geographic coordinates.

    Parameters
    ----------
    lat : ndarray
        Latitude values
    lon : ndarray
        Longitude values
    alpha : int, optional
        The smoothing coefficient for the RBF kernel, by default 1.0

    Returns
    -------
    torch.Tensor
        The RBF kernel matrix
    """
    lat_tensor = torch.tensor(lat).view(-1)
    lon_tensor = torch.tensor(lon).view(-1)

    # Create pos using meshgrid
    x, y = torch.meshgrid(lat_tensor, lon_tensor, indexing="ij")
    pos = torch.stack((x.flatten(), y.flatten()), dim=1)

    # Calculate pairwise distances
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    distances = torch.sum(diff**2, dim=-1)

    # Calculate kernel
    kernel = torch.exp(-distances / (2 * alpha**2))
    return torch.linalg.inv(kernel)


def get_kernel(data, config):
    lat = data.coords.variables["lat"].values.astype(np.float32)
    lon = data.coords.variables["lon"].values.astype(np.float32)
    kernel = get_gauss_kernel(lat, lon, config["rbf_alpha"]).to(config["device"])
    kernel = kernel.expand(config["bs"], *kernel.shape)
    return kernel


def stack_past_samples(tensor, n):
    new_shape = (tensor.shape[0] - n + 1, *tensor.shape[1:], n)
    new_strides = (*tensor.stride(), tensor.stride(0))
    return torch.as_strided(tensor, new_shape, new_strides, tensor.storage_offset())


def select_from_slice_for_batch(tensor, batch_size):
    i = tensor.shape[0] // batch_size
    num_elements = i * batch_size
    output = (
        tensor[:num_elements].view(i, batch_size, *tensor.shape[1:]).transpose(0, 1)[0]
    )
    if tensor.shape[0] % batch_size != 0:
        remaining = tensor[num_elements + 1]
        output = torch.cat((output, remaining.unsqueeze(0)), dim=0)
    return output


def compute_delta(tensor):
    stack_velocity_depth = tensor.shape[-1]
    t = torch.arange(0, stack_velocity_depth, 1).to(torch.float32)
    input_output_shaped = tensor.movedim(-1, 1)
    input = input_output_shaped.flatten(start_dim=2)
    coeffs = natural_cubic_spline_coeffs(t, input)
    spline = NaturalCubicSpline(coeffs)
    return spline.derivative(t[-1]).view(
        input_output_shaped.shape[0], *input_output_shaped.shape[2:]
    )


def optimize_velocity(tens, tens_delta, kernel, config):
    batch_size, *shape = tens.shape
    v_x = torch.randn(*tens.shape, device=config["device"], requires_grad=True)
    v_y = torch.randn(*tens.shape, device=config["device"], requires_grad=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([v_x, v_y], lr=config["lr"])

    for epoch in range(config["fitting_epoch"]):
        optimizer.zero_grad()
        u_y, u_x = torch.gradient(tens, dim=(1, 2))
        transport = v_x * u_x + v_y * u_y
        compression = tens * (
            torch.gradient(v_x, dim=1)[0] + torch.gradient(v_y, dim=2)[0]
        )
        adv = transport + compression

        kernel_v_x, kernel_v_y = (
            v_x.view(batch_size, -1, 1),
            v_y.view(batch_size, -1, 1),
        )

        kernel = kernel[:batch_size]
        v_x_kernel = torch.matmul(kernel_v_x.transpose(1, 2), kernel)
        v_y_kernel = torch.matmul(kernel_v_y.transpose(1, 2), kernel)

        final_x = torch.matmul(v_x_kernel, kernel_v_x).mean()
        final_y = torch.matmul(v_y_kernel, kernel_v_y).mean()

        vel_loss = criterion(tens_delta, adv) + config["regul_coeff"] * (
            final_x + final_y
        )
        vel_loss.backward()
        optimizer.step()

    v_x, v_y = v_x.detach(), v_y.detach()
    return torch.stack((v_x, v_y), dim=1)


def get_batch_velocity(batch, kernel, config):
    deltas = batch.apply(compute_delta, device=config["device"])
    datas = batch.apply(lambda x: x[:, :, :, -1], device=config["device"])
    keys = deltas.sorted_keys
    return TensorDict(
        {k: optimize_velocity(datas[k], deltas[k], kernel, config) for k in keys},
        batch_size=len(batch),
    )


def fit_velocity(period_data, kernel, config):
    s = config["vel"]["stacking"]
    new_batch_size = period_data.batch_size[0] - s + 1
    data_train = period_data.apply(
        lambda x: stack_past_samples(x, n=s), batch_size=[new_batch_size]
    )

    init_data_batch_size = data_train.batch_size[0] // config["pred_length"] + (
        data_train.batch_size[0] % config["pred_length"] > 0
    )
    init_data = data_train.apply(
        lambda x: select_from_slice_for_batch(x, batch_size=config["pred_length"]),
        batch_size=[init_data_batch_size],
    )

    dataloader = DataLoader(
        init_data,
        batch_size=config["vel"]["bs"],
        shuffle=False,
        collate_fn=lambda x: x,
        pin_memory=True,
    )

    velocities = [
        get_batch_velocity(batch, kernel, config["vel"]) for batch in tqdm(dataloader)
    ]

    return torch.cat(velocities, dim=0)


def get_hash(period_name, config):
    internal_config = {"period_name": period_name}
    internal_config.update(config["vel"])
    internal_config["freq"] = str(config["freq"]) + "H"
    internal_config["interval"] = config["periods"][period_name]
    internal_config["pred_length"] = config["pred_length"]
    internal_config.pop("device")

    import hashlib

    hash_object = hashlib.md5(str(internal_config).encode("utf-8"))
    return hash_object.hexdigest()


def get_period_velocity(period_name, period_data, kernel, config):
    param_hash = get_hash(period_name, config)
    velocities_save_path = f"./data/velocities/{period_name}"
    save_file = velocities_save_path + f"/velocities_{param_hash}.pt"
    os.makedirs(velocities_save_path, exist_ok=True)

    if os.path.exists(save_file):
        logging.info(f"Velocities for {period_name} loaded from cache")
        return torch.load(save_file)
    else:
        velocity = fit_velocity(period_data, kernel, config)
        torch.save(velocity, save_file)
        logging.info(f"Velocities for {period_name} computed and saved to cache")
        return velocity


def get_velocities(data, kernel, config):
    velocities = {}
    for period_name, period in data.items():
        velocities[period_name] = get_period_velocity(
            period_name, period, kernel, config
        )
    return velocities
