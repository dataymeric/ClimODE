import torch
from utils import get_constants


def get_day_and_season_embeddings(t):
    """Returns the day and season embeddings.
    TODO:
    [] Check if the embeding change **a lot** with bissextile year

    Parameters
    ----------
    t : Tensor (8760)=(365*24)
        Hours of the years

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
    day_in_years = t / 24  # 365 or 366
    hours_of_day = t % 24
    day_of_years = t // 24
    return torch.stack(
        (
            torch.sin(2 * torch.pi * hours_of_day),  # sin temporal embedding
            torch.cos(2 * torch.pi * hours_of_day),  # cos temporal embedding
            torch.sin(
                2 * torch.pi * day_of_years / day_in_years
            ),  # sin seasonal embedding
            torch.cos(
                2 * torch.pi * day_of_years / day_in_years
            ),  # cos seasonal embedding
        ),
        dim=1,
    )
    """
    Ce qui est fait dans le code du papier est un peu différent à mon gout, à cause du - pi/2
    t_emb = (t % 24).view(-1, 1, 1, 1, 1)
    sin_t_emb = torch.sin(torch.pi * t_emb / 12 - torch.pi / 2)
    cos_t_emb = torch.cos(torch.pi * t_emb / 12 - torch.pi / 2)
    sin_seas_emb = torch.sin(torch.pi * t_emb / (12 * 365) - torch.pi / 2)
    cos_seas_emb = torch.cos(torch.pi * t_emb / (12 * 365) - torch.pi / 2)
    => cat 
    
    Attention, c'est un autre calcul qui est fait dans les embedding pour 
    le PDE et la résolution de l'équation différentielle
    """


def get_localisation_embeddings(lat, lon):
    """Get localisation embeddings.

    Parameters
    ----------
    lat : torch.Tensor
        Latitude h \in [-90°, 90°]
    lon : torch.Tensor
        Longitude w \in [-180°, 180°]

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
        dim=1,
    )


