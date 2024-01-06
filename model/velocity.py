import torch


def get_gauss_kernel(lat, lon, alpha=1.0):
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
    x, y = torch.meshgrid(lat_tensor, lon_tensor, indexing='ij')
    pos = torch.stack((x.flatten(), y.flatten()), dim=1)

    # Calculate pairwise distances
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    distances = torch.sum(diff ** 2, dim=-1)

    # Calculate kernel
    kernel = torch.exp(-distances / (2 * alpha ** 2))
    return torch.linalg.inv(kernel)