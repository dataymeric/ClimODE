import torch
import xarray as xr


def get_constants(path):
    """Get constant spatial and time features.
    
    Parameters
    ----------
    path : string
        Path to load the constant spatial and times features

    Returns
    -------
    (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
        \[ \psi(c) = [\psi(h), \psi(w), \text{lsm}, \text{oro}] \]
    """
    constants = xr.open_mfdataset(path, combine='by_coords')
    oro = torch.tensor(constants["orography"].values)[(None,) * 2]
    lsm = torch.tensor(constants["lsm"].values)[(None,) * 2]
    lat2d = torch.tensor(constants['lat2d'].values)
    lon2d = torch.tensor(constants['lon2d'].values)
    return oro, lsm, lat2d, lon2d
