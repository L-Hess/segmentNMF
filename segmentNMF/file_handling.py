import numpy as np
import zarr
import nrrd
import copy

from scipy.stats import entropy

from zarr.indexing import BasicIndexer
from numcodecs import Blosc
from distributed import Lock

from scipy.ndimage import binary_dilation as binary_dilation_scipy

from scipy.ndimage import distance_transform_edt

import numpy as np
import zarr
import nrrd
import tempfile
import shutil

def create_temp_zarr(data):
    """
    Create a temporary Zarr array from the input data.

    Args:
        data (ndarray): The input data.

    Returns:
        zarr.core.Array: The Zarr array.

    """
    temp_dir = tempfile.mkdtemp()
    temp_file_path = f"{temp_dir}/data.zarr"
    zarr.save(temp_file_path, data)
    zarr_array = zarr.open(temp_file_path, mode='r')
    return zarr_array


def file_handler(file_path):
    """
    Handle different file types and load the data accordingly.

    Args:
        file_path (str): The path to the file.

    Returns:
        ndarray or zarr.core.Array: The loaded data.

    Raises:
        ValueError: If the file type is unsupported.

    """
    file_extension = file_path.split('.')[-1].lower()

    if file_extension == 'npy':
        # Load numpy file and create temporary Zarr array
        data = np.load(file_path)
        zarr_array = create_temp_zarr(data)
        return zarr_array

    elif file_extension == 'zarr':
        # Load zarr file (lazy loading)
        data = zarr.open(file_path, mode='r')
        return data

    elif file_extension == 'nrrd':
        # Load nrrd file and create temporary Zarr array
        data, header = nrrd.read(file_path)
        zarr_array = create_temp_zarr(data)
        return zarr_array

    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


def dilate_outer_layers(im, rad=[1, 1, 1], k: int = 1):
    """
    Dilate the outer layers of a binary image.

    Args:
        im (ndarray): The binary image.
        rad (list): The radius of the dilation in each dimension.
        k (int): The number of dilation iterations.

    Returns:
        ndarray: The dilated image.

    """
    for _ in range(k):
        inds = np.stack(np.where(im)).T
        im_eroded = copy.deepcopy(im)
        for ind in inds:
            window = tuple([slice(np.max([ind[i] - rad[i], 0]),
                                  np.min([ind[i] + rad[i] + 1, im.shape[i]]))
                            for i in range(len(im.shape))])
            if np.any(im[window] == 0):
                im_eroded[window] = 1
        if np.sum(im_eroded) == 0:
            im_eroded = im
        im = im_eroded
    return im


def neighborhood_by_dilation(S, segments_shape, k: int = 1, method: str = 'scipy'):
    """
    Expand the segmentation labels by dilation in each cell neighborhood.

    Args:
        S (ndarray): The segmentation labels.
        segments_shape (tuple): The shape of the segmented volumes.
        k (int): The number of dilation iterations.
        method (str): The method for dilation ('scipy' or 'stand').

    Returns:
        ndarray: The expanded segmentation labels.

    """
    if method == 'scipy':
        dilation_func = binary_dilation_scipy
    elif method == 'stand':
        dilation_func = dilate_outer_layers

    B = S > 0
    dilate_radius = 3

    N_cells = S.shape[1]
    for n_i in range(N_cells):
        B_i = B[:, n_i].reshape(segments_shape)
        for z in range(segments_shape[0]):
            B_iz = B_i[z, :, :]
            for d_i in range(dilate_radius):
                B_iz = dilation_func(B_iz, iterations=k)
            B_i[z, :, :] = B_iz
        B[:, n_i] = B_i.flatten()

    return B


def euclidean_distance_binary_mask(arr, spacing):
    """
    Calculate the Euclidean distance transform of a binary array.

    Args:
        arr (ndarray): The binary array.
        spacing (float or tuple): The voxel spacing in each dimension.

    Returns:
        ndarray: The distance transform.

    """
    # Calculate the distance transform of the binary array
    arr = arr * -1 + 1
    dist_transform = distance_transform_edt(arr, spacing)

    return dist_transform


def neighborhood_by_distance(S, segments_shape, spacing, max_dist, subsample=None):
    """
    Expand the segmentation labels by considering the Euclidean distance.

    Args:
        S (ndarray): The segmentation labels.
        segments_shape (tuple): The shape of the segmented volumes.
        spacing (float or tuple): The voxel spacing in each dimension.
        max_dist (float): The maximum distance for expansion.

    Returns:
        ndarray: The expanded segmentation labels.

    """
    B = S > 0

    if subsample is not None:
        B_res = B.reshape(*segments_shape, B.shape[1])
        B_res = B_res[::subsample[0], ::subsample[1], ::subsample[2]]
        B_res = B_res.reshape(np.prod(B_res.shape[:-1]), B.shape[-1])
    else:
        B_res = B

    for c_i in range(B.shape[1]):
        dists = euclidean_distance_binary_mask(B[:, c_i].reshape(segments_shape), spacing)
        Bi = (dists < max_dist)
        if subsample is not None:
            Bi = Bi[::subsample[0], ::subsample[1], ::subsample[2]]
        B_res[:, c_i] = Bi.reshape(B_res[:, c_i].shape)
    return B_res


def create_zarr(path, shape, chunks, dtype, chunk_locked=False, client=None):
    """
    Create a Zarr array with specified properties.

    Args:
        path (str): The path to the Zarr store.
        shape (tuple): The shape of the array.
        chunks (tuple): The chunk shape.
        dtype (dtype): The data type.
        chunk_locked (bool): Whether to use chunk-level locking (optional).
        client (dask.distributed.Client): The Dask client for distributed locking (optional).

    Returns:
        zarr.core.Array: The Zarr array.

    """
    compressor = Blosc(
        cname='zstd', clevel=4, shuffle=Blosc.BITSHUFFLE,
    )
    zarr_disk = zarr.open(
        path, 'w',
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
    )

    # this code is currently never used within CircuitSeeker
    # keeping it around in case a use case comes up
    if chunk_locked:
        indexer = BasicIndexer(slice(None), zarr_disk)
        keys = (zarr_disk._chunk_key(idx.chunk_coords) for idx in indexer)
        lock = {key: Lock(key, client=client) for key in keys}
        lock['.zarray'] = Lock('.zarray', client=client)
        zarr_disk = zarr.open(
            store=zarr_disk.store, path=zarr_disk.path,
            synchronizer=lock, mode='r+',
        )

    return zarr_disk


def vol_mutual_information(volume1, volume2):
    """
    Calculate the normalized correlation between two 3D volumes.

    Args:
        volume1 (ndarray): The first 3D volume.
        volume2 (ndarray): The second 3D volume.

    Returns:
        float: The normalized correlation value.

    """
    volume1[np.isclose(volume1, 1e-12)] = np.nan
    volume2[np.isclose(volume2, 1e-12)] = np.nan

    # Flatten the volumes
    flat_volume1 = volume1.flatten()
    flat_volume2 = volume2.flatten()

    # Remove NaN values and corresponding indices
    valid_indices = np.logical_and(~np.isnan(flat_volume1),
                                   ~np.isnan(flat_volume2))
    flat_volume1 = flat_volume1[valid_indices]
    flat_volume2 = flat_volume2[valid_indices]

    # Compute joint histogram
    bins = int(np.sqrt(len(flat_volume1)))
    hist_2d, _, _ = np.histogram2d(flat_volume1, flat_volume2, bins=bins)

    # Calculate marginal histograms
    hist_marginal_1 = np.sum(hist_2d, axis=1)
    hist_marginal_2 = np.sum(hist_2d, axis=0)

    # Normalize the histograms
    pxy = hist_2d / np.sum(hist_2d)
    px = hist_marginal_1 / np.sum(hist_marginal_1)
    py = hist_marginal_2 / np.sum(hist_marginal_2)

    # Replace infinity values with a large finite value
    pxy[np.isinf(pxy)] = 1e-100
    px[np.isinf(px)] = 1e-100
    py[np.isinf(py)] = 1e-100

    # Calculate mutual information
    mutual_info = np.nansum(pxy * np.log2(pxy / (np.outer(px, py) + np.finfo(float).eps)))

    # Normalize mutual information
    mutual_info /= np.sqrt(entropy(px, base=2) * entropy(py, base=2) + np.finfo(float).eps)

    return mutual_info
