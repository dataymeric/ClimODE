import torch
from tensordict import TensorDict
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline


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


def stack_past_samples(tensor, n):
    new_shape = (tensor.shape[0] - n + 1, *tensor.shape[1:], n)
    new_strides = (*tensor.stride(), tensor.stride(0))
    return torch.as_strided(tensor, new_shape, new_strides, tensor.storage_offset())


def select_from_slice_for_batch(tensor, batch_size):
    i = tensor.shape[0] // batch_size
    num_elements = i * batch_size
    output = tensor[:num_elements].view(i, batch_size, *tensor.shape[1:]).transpose(0, 1)[0]
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
    return spline.derivative(t[-1]).view(input_output_shaped.shape[0], *input_output_shaped.shape[2:])


def optimize_velocity(tens, tens_delta, kernel, device, velocity_fitting_epoch=100, lr=2, regularization_coeff=10e-7):
    batch_size, *shape = tens.shape
    v_x = torch.nn.Parameter(torch.randn(*tens.shape, device=device))
    v_y = torch.nn.Parameter(torch.randn(*tens.shape, device=device))

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([v_x, v_y], lr=lr)

    for epoch in range(velocity_fitting_epoch):
        optimizer.zero_grad()
        u_y, u_x = torch.gradient(tens, dim=(1, 2))
        transport = v_x * u_x + v_y * u_y
        compression = tens * (torch.gradient(v_x, dim=1)[0] + torch.gradient(v_y, dim=2)[0])
        adv = transport + compression

        kernel_v_x, kernel_v_y = v_x.view(batch_size, -1, 1), v_y.view(batch_size, -1, 1)

        kernel = kernel[:batch_size]
        v_x_kernel = torch.matmul(kernel_v_x.transpose(1, 2), kernel)
        v_y_kernel = torch.matmul(kernel_v_y.transpose(1, 2), kernel)

        final_x = torch.matmul(v_x_kernel, kernel_v_x).mean()
        final_y = torch.matmul(v_y_kernel, kernel_v_y).mean()

        vel_loss = criterion(tens_delta, adv.squeeze(dim=1)) + regularization_coeff * (final_x + final_y)
        vel_loss.backward()
        optimizer.step()

    v_x, v_y = v_x.detach(), v_y.detach()
    return torch.stack((v_x, v_y), dim=1).squeeze()

def get_batch_velocity(batch, kernel, device):
    deltas = batch.apply(compute_delta, device=device)
    datas = batch.apply(lambda x: x[:, :, :, -1], device=device)
    keys = deltas.sorted_keys
    return TensorDict({k: optimize_velocity(datas[k], deltas[k], kernel, device=device) for k in keys}, batch_size=len(batch))
