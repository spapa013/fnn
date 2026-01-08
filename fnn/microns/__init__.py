import os
import torch
import hashlib
import tempfile
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
from fnn.microns.build import network
from fnn.microns.load import params, units, unit_ids
from fnn.utils import logging

BASE_URL = "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/functional_data/foundation_model/"
MODEL_FILENAME = "foundational_model_weights_and_metadata_v1.zip"
MODEL_ORI_FILENAME = "foundation_model_weights_and_metadata_with_ori_training.zip"
README_FILENAME = "readme_v3.md"

md5_lookup = {
    MODEL_FILENAME: "58fcac4b31ad2902c81e339432cec787",
    MODEL_ORI_FILENAME: "4a6e0d5864c089193ad35db755d737f2",
}

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)


def download(url, file_path, chunk_size=8192, verbose=True):
    """
    Parameters
    ----------
    url : str
        source url
    file_path : os.PathLike
        destination file path
    chunk_size : int
        size of the chunks (in bytes) to read
    verbose : bool
        display download progress

    Returns
    -------
    str
        MD5 checksum of downloaded file as a hexadecimal string
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    size = int(response.headers.get("content-length", 0))
    md5 = hashlib.md5()

    with open(file_path, "wb") as file:
        with tqdm(total=size, unit="B", unit_scale=True, disable=not verbose) as bar:
            for chunk in response.iter_content(chunk_size):

                file.write(chunk)
                md5.update(chunk)
                bar.update(len(chunk))

    return md5.hexdigest()


def download_data(directory=None, verbose=True, filename=MODEL_FILENAME):
    """
    Parameters
    ----------
    directory : os.PathLike | None
        directory for model parameters and metadata. defaults to current working directory
    verbose : bool
        display download progress
    filename : str
        name of the model zip file to download
    """
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.info(f"Downloading model parameters and metadata to `{directory}`")
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "microns.zip")
        md5 = download(BASE_URL + filename, zip_path, verbose=verbose)
        md5_expected = md5_lookup[filename]
        assert md5 == md5_expected, f"md5 for downloaded file is {md5}, expected {md5_expected}"

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(directory)

        _ = download(BASE_URL + README_FILENAME, os.path.join(directory, "README.md"), verbose=verbose)


def scan(session, scan_idx, cuda=True, directory=None):
    """
    Parameters
    ----------
    session : int
        scan session
    scan_idx : int
        scan index
    cuda : bool
        use cuda if available
    directory : os.PathLike | None
        directory for model parameters and metadata. defaults to current working directory

    Returns
    -------
    fnn.model.networks.Visual
        predictive model of the experimental scan
    pd.DataFrame
        dataframe mapping readout ids to unit ids
    """
    directory = directory or os.getcwd()
    load = lambda f: f(session, scan_idx, directory)

    model = network(load(units))
    model.load_state_dict(load(params))

    if cuda and torch.cuda.is_available():
        model.to(device="cuda")

    return model, load(unit_ids)


def load_network_from_params(path_to_params, cuda=True):
    """
    Loads a neural network model from saved parameters.

    Args:
        path_to_params (str or Path): Path to the file containing the saved model parameters.
        cuda (bool, optional): If True and CUDA is available, moves the model to GPU. Defaults to True.

    Returns:
        torch.nn.Module: The loaded neural network model, on GPU if cuda is True and available, otherwise on CPU.

    Raises:
        FileNotFoundError: If the parameter file does not exist.
        KeyError: If required keys are missing in the parameter dictionary.
        RuntimeError: If there is an error loading the state dictionary.
    """

    path_to_params = Path(path_to_params)
    params = torch.load(path_to_params, map_location='cpu')
    n_units = params['readout.feature.weights.0'].shape[0]
    model = network(n_units)
    model.load_state_dict(params)
    if cuda and torch.cuda.is_available():
        return model.to(device="cuda")
    return model