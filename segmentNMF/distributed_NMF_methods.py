import numpy as np
import zarr
from ClusterWrap.decorator import cluster
from scipy.ndimage import find_objects
from scipy.spatial import cKDTree
from segmentNMF.file_handler import create_temp_zarr


#@cluster
def distributed_nmf(
    time_series_zarr,
    segments,
    time_series_spacing,
    segments_spacing,
    n_complete_cells_per_block=10,
    max_neighbor_distance=20.0,
    block_radius=4.0,
    cluster=None,
    cluster_kwargs={},
    **kwargs,
):
    """
    Run segmentNMF.NMF_methods.nmf_pytorch on a large volume.
    Computation is distributed over blocks run in parallel on a cluster.
    Compute blocks are the union of maximally inscribing parallelpipeds,
    where n_complete_cells_per_block cells are used per block (see Parameters).
    block_radius microns are added along the edge of each block (see Parameters).

    Parameters
    ----------
    time_series_zarr : zarr.core.Array
        A zarr array containing the time series data

    segments : numpy.ndarray (unsigned integer datatype)
        A numpy array containing the masks used to initialize the spatial
        components. Should be positive integer valued with each mask a unique integer
        value. Should be the same spatial domain as time_series_zarr, but can be
        sampled on a different voxel grid.

    time_series_spacing : numpy.ndarray, 1d (float datatype)
        The voxel spacing of the time_series_zarr grid

    segments_spacing : numpy.ndarray, 1d (float datatype)
        The voxel spacing of the segments grid

    n_complete_cells_per_block : int (default: 10)
        The number of neighboring cells used to define a compute block

    max_neighbor_distance : float (default: 20.0)
        The maximum distance in microns that two segment centers can be to
        still be combined in the same compute block. This helps prevent
        accidental construction of large compute blocks.

    block_radius : float (default: 4.0)
        The number of microns to extend each block

    cluster : ClusterWrap.cluster object (default: None)
        Only set if you have constructed your own static cluster. The default
        behavior is to construct a cluster for the duration of this function, then
        close it when the function is finished.

    cluster_kwargs : dict (default: {})
        Arguments passed to ClusterWrap.cluster
        If working with an LSF cluster, this will be
        ClusterWrap.janelia_lsf_cluster. If on a workstation this
        will be ClusterWrap.local_cluster.
        This is how distribution parameters are specified.

    kwargs : additional keyword arguments
        Passed to segmentNMF.NMF_methods.nmf_pytorch. This is how you control
        e.g. learning rates for each nmf run. See documentation for
        segmentNMF.NMF_methods.nmf_pytorch.

    Returns
    -------
    TODO XXX
    """

    # find segment boxes, get centers, get neighbors
    segment_ids = np.unique(segments)
    if segment_ids[0] == 0: segment_ids = segment_ids[1:]
    boxes = [box for box in find_objects(segments) if box is not None]
    centers = [[(x.start + x.stop)/2. for x in slc] for slc in boxes]
    centers = np.array(centers) * segments_spacing
    centers_tree = cKDTree(centers)
    neighbor_dists, neighbors = centers_tree.query(
        centers, k=n_complete_cells_per_block,
        distance_upper_bound=max_neighbor_distance,
    )

    # determine which segments will be unified to define compute blocks
    segment_unions = []
    segments_assigned = np.zeros(len(segment_ids), dtype=bool)
    while not np.all(segments_assigned):

        # first select smallest complete groups, then incomplete groups
        # with largest number of segments remaining
        neighbor_sums = np.sum(neighbor_dists, axis=1)
        if np.any(neighbor_sums < np.inf):
            selection = np.argmin(neighbor_sums)
            neighbor_ids = neighbors[selection]
        else:
            n_remaining = np.sum(neighbor_dists < np.inf, axis=1)
            selection = np.argmax(n_remaining)
            neighbor_ids = neighbors[selection]
            neighbor_ids = neighbor_ids[neighbor_dists[selection] < np.inf]

        # mark segments assigned, adjust_distances
        segment_unions.append(tuple(neighbor_ids))
        segments_assigned[neighbor_ids] = True
        neighbor_dists[neighbor_ids] = np.inf
        neighbor_dists[np.isin(neighbors, neighbor_ids)] = np.inf

    # compute box unions, expand by radius
    radius_voxels = np.ceil( block_radius / segments_spacing ).astype(int)
    def unify_and_expand(boxes):
        start = [min([x[i].start for x in boxes]) for i in range(segments.ndim)]
        stop = [max([x[i].stop for x in boxes]) for i in range(segments.ndim)]
        start = [max(0, x-r) for x, r in zip(start, radius_voxels)]
        stop = [min(s, x+r) for x, r, s in zip(stop, radius_voxels, segments.shape)]
        return tuple(slice(x, y) for x, y in zip(start, stop))
    box_unions = [unify_and_expand(x) for x in segment_unions]

    # compute time series crops
    sampling_ratio = segments_spacing / time_series_spacing
    def ts_box_from_seg_box(box):
        start = [int(np.floor(x.start*r)) for x, r in zip(box, sampling_ratio)]
        stop = [int(np.ceil(x.stop*r)) for x, r in zip(box, sampling_ratio)]
        return tuple(slice(x, y) for x, y in zip(start, stop))
    time_series_crops = [ts_box_from_seg_box(box) for box in box_unions]

    # compute segments crops (not the same as box_unions due to floor/ceil above)
    sampling_ratio = (1. / sampling_ratio)
    def seg_box_from_ts_box(box):
        start = [int(np.floor(x.start*r)) for x, r in zip(box, sampling_ratio)]
        stop = [int(np.floor(x.stop*r)) for x, r in zip(box, sampling_ratio)]
        return tuple(slice(x, y) for x, y in zip(start, stop))
    segments_crops = [seg_box_from_ts_box(box) for box in time_series_crops]

    # ensure segments are a zarr array
    segments = create_temp_zarr(segments)


    # the function to map on every block
    def nmf_per_block(segment_ids, time_series_crop, segments_crop):


    
