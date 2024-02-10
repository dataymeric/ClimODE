import torch


def get_day_and_season_embeddings(day_of_year_ratio, hour_of_day):
    """Returns the day and season embeddings.

    Parameters
    ----------
    day_of_year_ratio : torch.Tensor
        The day of the year divided by the number of days in the specified year.
    hour_of_day : torch.Tensor
        The hour of the day.

    Returns
    -------
    torch.Tensor
        A tensor containing the day and season embeddings.
        The tensor has shape (8760, 4), where each row represents a day and each column represents an embedding.
        The embeddings are as follows:
        - Column 0: sin temporal embedding
        - Column 1: cos temporal embedding
        - Column 2: sin seasonal embedding
        - Column 3: cos seasonal embedding
    """
    hours_of_day = hour_of_day % 24
    return torch.stack(
        (
            torch.sin(2 * torch.pi * hours_of_day),  # sin temporal embedding
            torch.cos(2 * torch.pi * hours_of_day),  # cos temporal embedding
            torch.sin(2 * torch.pi * day_of_year_ratio),  # sin seasonal embedding
            torch.cos(2 * torch.pi * day_of_year_ratio),  # cos seasonal embedding
        ),
        dim=1,
    )


def get_localisation_embeddings(lat, lon):
    """Get localisation embeddings.
    It's what is done in the forward loop.

    Parameters
    ----------
    lat : torch.Tensor
        Latitude h \in [-90°, 90°]
        (32, 64, 1)
    lon : torch.Tensor
        Longitude w \in [-180°, 180°].
        (32, 64, 1)

    Returns
    -------
    torch.Tensor (32, 64, 6)
    \[
        \psi(x) = \psi(h,w) = [\{\sin, \cos\} \times \{h, w}, \sin(h) \cos(w), \sin(h) \sin(w)
    \]
    """
    # Converting to radians
    cos_lat_map = torch.cos(lat * torch.pi / 180)
    cos_lon_map = torch.cos(lon * torch.pi / 180)
    sin_lat_map = torch.sin(lat * torch.pi / 180)
    sin_lon_map = torch.sin(lon * torch.pi / 180)
    return torch.cat(
        [
            cos_lat_map,  # cos(h)
            cos_lon_map,  # cos(w)
            sin_lat_map,  # sin(h)
            sin_lon_map,  # sin(w)
            sin_lat_map * cos_lon_map,  # sin(h) cos(w)
            sin_lat_map * sin_lon_map,  # sin(h) sin(w)
        ],
        dim=-1,
    )


def get_time_localisation_embeddings(
    day_of_year_ratio, hour_of_day, lat, lon, lsm, oro
):
    """Get join time-localisation embeddings $\psi(x, t)$ for every $t$ in the time step range.
    Embedding are in the same order as in the paper:
    * Day and season $\psi(t)$: (nb_time_step, 4)
    * Localisation $\psi(x)$: (32, 64, 6)
    * Time and position $\psi(t) \\times \psi(x)$: (nb_time_step, 32, 64, 24)
    * Latitude: (32, 64, 1)
    * Longitude: (32, 64, 1)
    * Land sea mask: (32, 64, 1)
    * Orography: (32, 64, 1)

    Parameters
    ----------
    day_of_year_ratio : torch.Tensor
        The day of the year divided by the number of days in the specified year.
    hour_of_day : torch.Tensor
        The hour of the day.
    lat : torch.Tensor
        latitude in 1D (32)
    lon : torch.Tensor
        longitude in 1D (64)
    lsm : torch.Tensor
        Land sea mask constant (32, 64, 1)
    oro : torch.Tensor
        Orography constant (32, 64, 1)

    Returns
    -------
    torch.Tensor
        (nb_time_step, 32, 64, 24)
    """

    def get_time_pos_embedding(day_seas_emb, loc_emb):
        """Create 6 new dimension in `final_out` for each dimension of `day_seas_emb`.
        * First repeat interleave `day_seas_emb` 6 times
        * Then repeat `loc_emb` 4 times
        * Multiply the two tensors
        So the 6 first dimensions of the final output are the 6 dimension of loc_emb multiplied by the first dimension of day_seas_emb.
        The 6 next dimensions of the final output are the 6 dimension of loc_emb multiplied by the second dimension of day_seas_emb.
        ect...

        Parameters
        ----------
        day_seas_emb : torch.Tensor
            (nb_time_step, 4])
        loc_emb : torch.Tensor
            (nb_time_step, 32, 64, 6)

        Returns
        -------
        torch.Tensor
            (nb_time_step, 32, 64, 38)
        """
        day_seas_emb = day_seas_emb.repeat_interleave(
            6, dim=-1
        )  # (nb_time_step, 32, 64, 4) -> (nb_time_step, 32, 64, 24)
        loc_emb = loc_emb.repeat(1, 1, 1, 4)  # (nb_time_step, 32, 64, 24)
        return day_seas_emb * loc_emb

    def add_time_step_dim(x):
        """
        (32,64,n) -> (nb_time_step, 32, 64, n)
        """
        return x.view(1, 32, 64, -1).expand(nb_time_step, 32, 64, -1)

    nb_time_step = len(hour_of_day)
    lat = lat.view(32, 1, 1).expand(32, 64, 1)
    lon = lon.view(1, 64, 1).expand(32, 64, 1)
    loc_emb = get_localisation_embeddings(lat, lon)  # (32, 64, 6)
    day_seas_emb = get_day_and_season_embeddings(
        day_of_year_ratio, hour_of_day
    )  # (nb_time_step, 4)

    # Preparing for localization and time embeddings combination
    loc_emb = loc_emb.expand(
        nb_time_step, 32, 64, 6
    )  # (32, 64, 6) -> (nb_time_step, 32, 64, 6)
    day_seas_emb = day_seas_emb.view(nb_time_step, 1, 1, -1).expand(
        nb_time_step, 32, 64, -1
    )  # day_seas_emb: (nb_time_step, 4) -> (nb_time_step, 32, 64, 4)
    return torch.cat(
        [
            day_seas_emb,  # (nb_time_step, 32, 64, 4)
            loc_emb,  # (nb_time_step, 32, 64, 6)
            get_time_pos_embedding(day_seas_emb, loc_emb),  # (nb_time_step, 32, 64, 24)
            add_time_step_dim(lat),  # (32,64,1) -> (nb_time_step, 32, 64, 1)
            add_time_step_dim(lon),  # (32,64,1) -> (nb_time_step, 32, 64, 1)
            add_time_step_dim(lsm),  # (32,64,1) -> (nb_time_step, 32, 64, 1)
            add_time_step_dim(oro),  # (32,64,1) -> (nb_time_step, 32, 64, 1)
        ],
        dim=-1,
    )
