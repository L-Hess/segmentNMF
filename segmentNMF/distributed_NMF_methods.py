import os, tempfile
import numpy as np
import zarr
from ClusterWrap.decorator import cluster
from scipy.ndimage import find_objects
from scipy.spatial import cKDTree
from segmentNMF.file_handling import create_temp_zarr
from segmentNMF.neighborhood import neighborhood_by_weights
from segmentNMF.NMF_methods import nmf


@cluster
def distributed_nmf(
    time_series_zarr,
    segments,
    time_series_spacing,
    segments_spacing,
    max_complete_cells_per_block=10,
    max_neighbor_distance=20.0,
    block_radius=4.0,
    time_series_baseline=None,
    neighborhood_sigma=2.0,
    temporary_directory=None,
    cluster=None,
    cluster_kwargs={},
    **kwargs,
):
    """
    Run segmentNMF.NMF_methods.nmf_pytorch on a large volume.
    Computation is distributed over blocks run in parallel on a cluster.
    Compute blocks are the union of maximally inscribing parallelpipeds,
    where max_complete_cells_per_block cells are used per block (see Parameters).
    block_radius microns are added along the edge of each block (see Parameters).

    Parameters
    ----------
    time_series_zarr : zarr.core.Array
        A zarr array containing the time series data. Time is assumed to be the
        first axis. Z (or the slice/anisotropic dimension) is assumed to be the
        second axis.

    segments : numpy.ndarray (unsigned integer datatype)
        A numpy array containing the masks used to initialize the spatial
        components. Should be positive integer valued with each mask a unique integer
        value. Should be the same spatial domain as time_series_zarr, but can be
        sampled on a different voxel grid. Z (or the slice/anisotropic dimension) is
        assumed to be the first axis.

    time_series_spacing : numpy.ndarray, 1d (float datatype)
        The voxel spacing of the time_series_zarr grid

    segments_spacing : numpy.ndarray, 1d (float datatype)
        The voxel spacing of the segments grid

    max_complete_cells_per_block : int (default: 10)
        The number of neighboring cells used to define a compute block

    max_neighbor_distance : float (default: 20.0)
        The maximum distance in microns that two segment centers can be to
        still be combined in the same compute block. This helps prevent
        accidental construction of large compute blocks.

    block_radius : float (default: 4.0)
        The number of microns to extend each block

    time_series_baseline : float (default: None)
        A scalar baseline value to subtract from the time series. Ignored if None.

    neighborhood_sigma : float (default: 2.0)
        Spatial component learning rates are scaled by weights that enforce
        the prior given structure. The weight function is:
        np.exp( -dist / neighborhood_sigma ) where dist is the distance of
        a location in the spatial component from the boundary of the given
        segment. Larger values allow spatial components to grow more.

    temporary_directory : string (default: None)
        Temporary files are created during nmf. The temporary files will be
        in their own folder within the `temporary_directory`. The default is the
        current directory. Temporary files are removed if the function completes
        successfully.

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
        centers, k=max_complete_cells_per_block,
        distance_upper_bound=max_neighbor_distance,
    )

    # TODO: THIS WHILE LOOP TAKES ABOUT 10 MINUTES, WE DON'T WANT THE CLUSTER WAITING
    # determine which segments will be unified to define compute blocks
    segment_unions = []
    segments_assigned = np.zeros(len(segment_ids), dtype=bool)
    temp_iii = 0
    while not np.all(segments_assigned) and temp_iii < 10:

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
        temp_iii += 1

    # compute box unions, expand by radius
    radius_voxels = np.ceil( block_radius / segments_spacing ).astype(int)
    def unify_and_expand(indices):
        _boxes = [boxes[i] for i in indices]
        start = [min([x[i].start for x in _boxes]) for i in range(segments.ndim)]
        stop = [max([x[i].stop for x in _boxes]) for i in range(segments.ndim)]
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

    # convert segment union indices to segment ids
    segment_unions = [tuple(segment_ids[i] for i in x) for x in segment_unions]

    # ensure segments are a zarr array
    # XXX tried segmentNMF.file_handling.create_temp_zarr but dask was not
    #     able to serialize/deserialize that zarr array for some reason
    segments_zarr = segments
    if not isinstance(segments_zarr, zarr.Array):
        temporary_directory = tempfile.TemporaryDirectory(
            prefix='.', dir=temporary_directory or os.getcwd(),
        )
        zarr_chunks = (128,) * segments.ndim
        segments_zarr_path = temporary_directory.name + '/segments.zarr'
        synchronizer = zarr.ThreadSynchronizer()
        segments_zarr = zarr.open(
            segments_zarr_path, 'w',
            shape=segments.shape,
            chunks=zarr_chunks,
            dtype=segments.dtype,
            synchronizer=synchronizer,
        )
        segments_zarr[...] = segments


    # the function to map on every block
    def nmf_per_block(segment_ids, time_series_crop, segments_crop):

        # read the time series data carefully, don't let zarr cache whole blocks
        frames = time_series_zarr.shape[0]
        ts = (frames,) + tuple(x.stop - x.start for x in time_series_crop)
        time_series = np.empty(ts, dtype=time_series_zarr.dtype)
        for iii in range(frames):
            time_series[iii] = time_series_zarr[(slice(iii, iii+1),) + time_series_crop]

        # read the segments, should be small data can read easily
        segments = segments_zarr[segments_crop]

        # ensure the segments of interest are the first components
        n_segments = len(segment_ids)
        segments[segments > 0] += n_segments
        for iii, segment_id in enumerate(segment_ids):
            segments[segments == segment_id + n_segments] = iii + 1

        # format the time series data
        V = time_series.astype(np.float32).reshape((ts[0], np.prod(ts[1:]))).T
        if time_series_baseline is not None:
            V = np.maximum(0, V - time_series_baseline)

        # format the spatial components
        labels = np.unique(segments)
        if labels[0] == 0: labels = labels[1:]
        n_labels = len(labels)

        pf = int(np.round(sampling_ratio[0]))  # projection factor
        S = np.empty((np.prod(ts[1:]), n_labels), dtype=np.float32)
        for i, n_i in enumerate(labels):
            comp = np.zeros(ts[1:], dtype=S.dtype)
            for j in range(comp.shape[0]):
                comp[j] = np.max(segments[j*pf:(j+1)*pf] == n_i, axis=0)
            S[:, i] = comp.reshape(np.prod(ts[1:]))

        # format search neighborhoods
        B = neighborhood_by_weights(
            S, segments_shape=ts[1:],
            sigma=neighborhood_sigma,
            spacing=time_series_spacing,
        )

        # format the temporal components
        H = np.zeros((n_labels, ts[0]), dtype=np.float32)

        # TEMP XXX
        S_init = np.copy(S)

        # run nmf
        S_res, H_res, objectives = nmf(
            V, S, H, B,
            estimate_noise_component=False,
            **kwargs,
        )

        # XXX TEMP XXX
        return time_series, segments, S_init, S_res, H_res, objectives

        # TODO crop out and return only the segments we care about


    # map nmf function on all blocks
    futures = cluster.client.map(
        nmf_per_block,
        segment_unions,
        time_series_crops,
        segments_crops,
    )

    # TODO: fuse all results into single array return with all necessary info

    # XXX TEMP XXX
    return cluster.client.gather(futures)

    
