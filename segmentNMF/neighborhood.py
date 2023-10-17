"""Functions for computing neighborhoods of spatial components"""

import numpy as np
import copy

from scipy.ndimage import binary_dilation as binary_dilation_scipy
from scipy.ndimage import distance_transform_edt


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


def distance_transform_edt_binary(arr, spacing):
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


def neighborhood_by_weights(S, segments_shape, spacing, sigma, subsample=None):

    """
    Expand the segmentation labels with weights that fall off

    Args:
        S (ndarray): The segmentation labels.
        segments_shape (tuple): The shape of the segmented volumes.
        spacing (float or tuple): The voxel spacing in each dimension.
        sigma (float): Standard deviation of exponential fall off for weights
        subsample (iterable): Factors by which to skip sample each axis

    Returns:
        ndarray: The components spatial weights

    """
    B = (S > 0).astype(np.float32)

    if subsample is not None:
        B_res = B.reshape(*segments_shape, B.shape[1])
        B_res = B_res[::subsample[0], ::subsample[1], ::subsample[2]]
        B_res = B_res.reshape(np.prod(B_res.shape[:-1]), B.shape[-1])
    else:
        B_res = B

    for c_i in range(B.shape[1]):
        dists = distance_transform_edt_binary(B[:, c_i].reshape(segments_shape), spacing)
        Bi = np.exp( -dists / sigma )
        if subsample is not None:
            Bi = Bi[::subsample[0], ::subsample[1], ::subsample[2]]
        B_res[:, c_i] = Bi.reshape(B_res[:, c_i].shape)
    return B_res


def neighborhood_by_distance(S, segments_shape, spacing, max_dist, subsample=None):
    """
    Expand the segmentation labels by considering the Euclidean distance.

    Args:
        S (ndarray): The segmentation labels.
        segments_shape (tuple): The shape of the segmented volumes.
        spacing (float or tuple): The voxel spacing in each dimension.
        max_dist (float): The maximum distance for expansion.
        subsample (iterable): Factors by which to skip sample each axis

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
        dists = distance_transform_edt_binary(B[:, c_i].reshape(segments_shape), spacing)
        Bi = (dists < max_dist)
        if subsample is not None:
            Bi = Bi[::subsample[0], ::subsample[1], ::subsample[2]]
        B_res[:, c_i] = Bi.reshape(B_res[:, c_i].shape)
    return B_res

