import torch
import pandas as pd
import xarray as xr
# lat = torch.tensor(train_raw_data["lat"].values) * torch.pi / 180
# cos_h = torch.cos(lat)
# latitude_weight = cos_h / cos_h.mean()
# latitude_weight = latitude_weight.view(1, 1, 1, 32, 1).expand_as(mean_pred)


def batch_rmse(mean_pred, mean_true, latitude_weight):
    loss = latitude_weight * (mean_pred - mean_true) ** 2
    # Mean on the latitude and longitude axis
    loss = loss.mean(dim=(-1, -2))
    # Mean on the batch axis, leaving only the variable axis
    loss = torch.sqrt(loss).mean(dim=(0, 1))
    return loss


def predict_to_zarr(model, test_loader, raw_data, config, variables_time_dependant, periods):
    """Might be splited into smaller function
    """
    # Getting original index
    data_test = raw_data.sel(time=periods["test"][:-5][:: config["pred_length"]])[
        variables_time_dependant
    ]
    timedelta = [
        pd.Timedelta(i * config["freq"], "h") for i in range(config["pred_length"])
    ]
    lat = data_test.get_index("lat")
    lon = data_test.get_index("lon")
    time = data_test.get_index("time")
    idx = {"latitude": lat, "longitude": lon, "time": time, "timedelta": timedelta}

    # Prediction on the test set
    l = []  # noqa: E741
    for data, vel, t in test_loader:
        data = data.to(config["device"])
        vel = vel.to(config["device"])

        mean, std = model(data, vel, t)
        l.append(mean.cpu().detach())
    data = torch.cat(l, dim=0).view(32, 64, 364, 8, 5).numpy()

    # Creating xarray dataset
    dims = ("latitude", "longitude", "time", "timedelta")
    d = {
        "2m_temperature": (dims, data[..., 0]),
        "temperature": (dims, data[..., 1]),
        "geopotential": (dims, data[..., 2]),
        "10m_u_component_of_wind": (dims, data[..., 3]),
        "10m_v_component_of_wind": (dims, data[..., 4]),
    }
    return xr.Dataset(d, coords=idx)
    
