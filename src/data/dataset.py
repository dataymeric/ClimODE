import torch
from torch.utils.data import Dataset


class Forcasting_ERA5Dataset(Dataset):
    def __init__(self, dataset, velocities, pred_length):
        """From the weather at one time step, we want to predict the weather at the next `nb_timestep` time steps.

        Parameters
        ----------
        dataset : torch.Tensor
            The dataset
        velocities : torch.Tensor
            The velocities of the dataset
        pred_length : int
            Number of timestep to predict
        """
        # Load and preprocess your data here
        self.data = dataset[:-1]
        self.velocities = velocities[:-1]

        self.pred_length = pred_length

    def __len__(self):
        return len(self.velocities)

    def __getitem__(self, index):
        data = self.data[index * self.pred_length : (index + 1) * self.pred_length]
        vel = self.velocities[index]
        return data, vel, int(index * self.pred_length)


def collate_fn(batch):
    data2 = [i[0] for i in batch]
    velocities = [i[1] for i in batch]
    time = [i[2] for i in batch]
    try:
        data = torch.stack(list(torch.stack(data2).values()), dim=2)
    except:
        print(data2)
    velocities = torch.stack(list(torch.stack(velocities).values()), dim=1)
    time = torch.tensor(time)

    return data, velocities, time
