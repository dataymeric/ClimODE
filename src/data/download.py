"""
This module provides functionality to download ERA5 data from specified URLs and extract 
it into a local directory.
"""
import os
from urllib.parse import parse_qs, urlparse
from zipfile import ZipFile

import requests
from tqdm import tqdm


def get_filename_from_url(url):
    """Extract filename from URL."""
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    if "files" in query_params:
        return query_params["files"][0]
    else:
        return os.path.basename(parsed_url.path)


def get_folder_path_from_url(url):
    """Extract the folder path from the URL."""
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    if "path" in query_params:
        folder_path = query_params["path"][0].strip("/").split("/", 1)[-1]
        return folder_path
    else:
        return ""


if __name__ == "__main__":
    # Target path for the data
    data_path = "/tmp/era5_data"
    # URLs for the files
    urls = [
        "https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg%2Fconstants&files=constants_5.625deg.nc",
        "https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg%2Fgeopotential_500&files=geopotential_500_5.625deg.zip",
        "https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg%2Ftemperature_850&files=temperature_850_5.625deg.zip",
        "https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg%2F2m_temperature&files=2m_temperature_5.625deg.zip",
        "https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg%2F10m_u_component_of_wind&files=10m_u_component_of_wind_5.625deg.zip",
        "https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg%2F10m_v_component_of_wind&files=10m_v_component_of_wind_5.625deg.zip",
        # "https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg%2Fu_component_of_wind&files=u_component_of_wind_5.625deg.zip",
        # "https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg%2Fv_component_of_wind&files=v_component_of_wind_5.625deg.zip",
        # "https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg%2Frelative_humidity&files=relative_humidity_5.625deg.zip",
        # "https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg%2Fspecific_humidity&files=specific_humidity_5.625deg.zip",
    ]

    # Loop through each URL and download/unzip
    for url in urls:
        folder_path = get_folder_path_from_url(url)
        filename = get_filename_from_url(url)
        tmp_download_path = os.path.join(data_path, folder_path, filename)
        tmp_extract_path = os.path.join(data_path, folder_path)

        # Disable SSL certificate verification
        response = requests.get(url, stream=True, verify=False)

        if response.status_code == 200:
            total_size = int(response.headers.get("content-length", 0))
            os.makedirs(os.path.dirname(tmp_download_path), exist_ok=True)
            with open(tmp_download_path, "wb") as file, tqdm(
                desc=filename,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    file.write(data)
                    bar.update(len(data))

            if filename.endswith(".zip"):
                with ZipFile(tmp_download_path, "r") as zip_file:
                    os.makedirs(tmp_extract_path, exist_ok=True)
                    zip_file.extractall(path=tmp_extract_path)

                # Remove the zip file after extraction
                os.remove(tmp_download_path)

            print(f"Downloaded and extracted to {data_path}/{folder_path}: {filename}")
        else:
            print(f"Failed to download: {filename}")
