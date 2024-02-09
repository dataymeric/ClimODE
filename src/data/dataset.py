from torch.utils.data import Dataset


class Forcasting_ERA5Dataset(Dataset):
    def __init__(self, dataset, nb_timestep=8):
        """From the weather at one time step, we want to predict the weather at the next `nb_timestep` time steps.

        Parameters
        ----------
        dataset : _type_
            The dataset
        nb_timestep : int
            Number of timestep to predict
        """
        # Load and preprocess your data here
        self.data = dataset
        self.nb_timestep = nb_timestep

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        # On prends les nb_timestep inclus (+1) time steps suivantes
        # sans inclure la donnée courante (+1)
        y = self.data[index + 1 : index + self.nb_timestep + 1]
        return x, y
