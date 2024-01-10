import os, tempfile
import numpy as np
import zarr
import time
import glob
from ClusterWrap.decorator import cluster as cluster_decorator
from scipy.ndimage import find_objects
from scipy.spatial import cKDTree
from segmentNMF.neighborhood import neighborhood_by_weights
from segmentNMF.NMF_methods import nmf, nmf_pytorch


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
    use_gpu=False,
    n_output_segments=None,
    reconstruction_path=None,
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

    use_gpu : bool (default: False)
        If False blocks are submitted to segmentNMF.NMF_methods.nmf function.
        If True blocks are submitted to segmentNMF.NMF_methods.nmf_pytorch function.
        The pytorch function can still run on a cpu, but requires more memory and
        is slower. You should only set this to True if you have a cuda compatible
        gpu available on your machine or your workers.

    n_output_segments : int (default: None)
        The number of rows in the output. If None, then the maximum integer value
        found in the `segments` array is used. If this is not None then it must
        be greater than the maximum integer value found in `segments`.

    reconstruction_path : string (default: None)
        If not None, this should be a path to a place on disk where a zarr
        array can be created to contain the reconstruction:
        space_components @ time_components.

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
    time_components : 2d numpy.ndarray
        The temporal component for every segment. This is an NxT array
        where T is the number of time points (or frames) in the time_series_zarr
        data. If `N_outout_segments` is None then N is equal to the maximum
        integer value found in the `segments` array. If `n_output_segments` is
        an integer then N is that number. Integer values not present in
        the `segments` array are NaN value for all T.
    """

    # get all segment ids
    print('GETTING UNIQUE SEGMENT IDS')
    segment_ids = np.unique(segments)
    if segment_ids[0] == 0: segment_ids = segment_ids[1:]

    # ensure output format will make sense
    if n_output_segments is not None:
        error_message = "n_output_segments must exceed maximum integer "
        error_message += "value in segments array\n. n_output_segments: "
        error_message += f"{n_output_segments} np.max(segments): {segment_ids[-1]}"
        assert (n_output_segments >= segment_ids[-1]), error_message
    else:
        n_output_segments = segment_ids[-1]

    # find segment boxes
    print('GETTING BOXES FOR EVERY SEGMENT')
    boxes = [box for box in find_objects(segments) if box is not None]

    # TODO: this is a hack to deal with bad segments, need a more proper solution
    bad_box_ids = []
    for iii, box in enumerate(boxes):
        sizes = np.array([x.stop - x.start for x in box])
        if np.any(sizes > 40): bad_box_ids.append(iii)
    segment_ids = [x for i, x in enumerate(segment_ids) if i not in bad_box_ids]
    boxes = [x for i, x in enumerate(boxes) if i not in bad_box_ids]

    # get centers, get neighbors
    centers = [[(x.start + x.stop)/2. for x in slc] for slc in boxes]
    centers = np.array(centers) * segments_spacing
    centers_tree = cKDTree(centers)
    neighbor_dists, neighbors = centers_tree.query(
        centers, k=max_complete_cells_per_block,
        distance_upper_bound=max_neighbor_distance,
    )

    # determine which segments will be unified to define compute blocks
    print('DETERMINING BLOCK STRUCTURE')
    print("this step can take a while, but don't worry, the cluster has not been constructed yet")
    segment_unions = []
    segments_assigned = np.zeros(len(segment_ids), dtype=bool)
    while not np.all(segments_assigned):

        # first select smallest complete groups, then incomplete groups
        # with largest number of unassigned segments remaining
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
    print('COMPUTING ALL CROPS')
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
    def seg_to_ts_box(box):
        start = [int(np.floor(x.start*r)) for x, r in zip(box, sampling_ratio)]
        stop = [int(np.ceil((x.stop-1)*r))+1 for x, r in zip(box, sampling_ratio)]
        return tuple(slice(x, y) for x, y in zip(start, stop))
    time_series_crops = [seg_to_ts_box(box) for box in box_unions]

    # compute segments crops
    sampling_ratio = time_series_spacing / segments_spacing
    radius = np.round((sampling_ratio - 1) / 2).astype(int)
    def ts_to_seg_box(box):
        start = [int(np.round(x.start*r)) for x, r in zip(box, sampling_ratio)]
        stop = [int(np.round((x.stop-1)*r))+1 for x, r in zip(box, sampling_ratio)]
        start = [max(0, x-r) for x, r in zip(start, radius)]
        stop = [min(s, x+r) for x, r, s in zip(stop, radius, segments.shape)]
        return tuple(slice(x, y) for x, y in zip(start, stop))
    segments_crops = [ts_to_seg_box(box) for box in time_series_crops]

    # convert segment union indices to segment ids
    segment_unions = [tuple(segment_ids[i] for i in x) for x in segment_unions]

    # make tempfile if we need it
    A = not isinstance(segments, zarr.Array)
    B = reconstruction_path is not None
    if A or B:
        temporary_directory = tempfile.TemporaryDirectory(
            prefix='.', dir=temporary_directory or os.getcwd(),
        )
        temp_dir_path = temporary_directory.name

    # ensure segments are a zarr array
    # tried segmentNMF.file_handling.create_temp_zarr but dask was not
    # able to serialize/deserialize that zarr array for some reason
    segments_zarr = segments
    if not isinstance(segments_zarr, zarr.Array):
        zarr_chunks = (128,) * segments.ndim
        segments_zarr_path = temp_dir_path + '/segments.zarr'
        segments_zarr = zarr.open(
            segments_zarr_path, 'w',
            shape=segments.shape,
            chunks=zarr_chunks,
            dtype=segments.dtype,
            synchronizer=zarr.ThreadSynchronizer(),
        )
        segments_zarr[...] = segments

    # create reconstruction zarr array
    if reconstruction_path is not None:
        reconstruction_zarr = zarr.open(
            reconstruction_path, 'w',
            shape=time_series_zarr.shape,
            chunks=time_series_zarr.chunks,
            dtype=np.float32,
            synchronizer=zarr.ThreadSynchronizer(),
        )

        # create weights array for reconstruction averaging
        reconstruction_weights = np.zeros(time_series_zarr.shape[1:], dtype=np.float32)
        for crop in time_series_crops:
            reconstruction_weights[crop] += 1
        reconstruction_weights[reconstruction_weights == 0] = 1
        reconstruction_weights = reconstruction_weights**-1

        # store reconstruction weights as zarr
        rw_path = temp_dir_path + '/reconstruction_weights.zarr'
        reconstruction_weights_zarr = zarr.open(
            rw_path, 'w',
            shape=reconstruction_weights.shape,
            chunks=time_series_zarr.chunks[1:],
            dtype=reconstruction_weights.dtype,
            synchronizer=zarr.ThreadSynchronizer(),
        )
        reconstruction_weights_zarr[...] = reconstruction_weights


    # the function to map on every block
    def nmf_per_block(
        segment_ids,
        time_series_crop,
        segments_crop,
    ):

        print(f'SEGMENT IDS: {segment_ids}\nTIME_SERIES_CROP: {time_series_crop}', flush=True)

        # read segments and time series
        time_series = time_series_zarr[(slice(None),) + time_series_crop]
        ts = time_series.shape
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
        #    get all labels
        labels = np.unique(segments)
        if labels[0] == 0: labels = labels[1:]
        n_labels = len(labels)

        #    determine relationship between segments and time series planes
        pf = int(np.round(sampling_ratio[0]))  # projection_factor
        weights = np.abs(np.arange(-(pf//2), pf//2+1)) * segments_spacing[0]
        weights = np.exp(-weights / neighborhood_sigma)
        plane_weights = np.empty((pf,) + segments.shape[1:], dtype=np.float32)
        plane_weights[...] = weights.reshape((pf,) + (1,) * (segments.ndim - 1))

        #    make a component for each label
        #    note: segments crop already contains radius to account for projections
        #          i.e. segments crop is a little bigger than time series crop
        S = np.empty((np.prod(ts[1:]), n_labels), dtype=np.float32)
        includes_first_plane = (time_series_crop[0].start == 0)
        includes_last_plane = (time_series_crop[0].start >= 0)
        for i, n_i in enumerate(labels):
            start = 0
            comp = np.zeros(ts[1:], dtype=S.dtype)
            for j in range(comp.shape[0]):
                seg_crop = slice(start, start + pf)
                w_crop = slice(None)
                if includes_first_plane and j == 0:
                    seg_crop = slice(0, pf//2 + 1)
                    w_crop = slice(-(pf//2 + 1), None)
                elif includes_last_plane and j == comp.shape[0]-1:
                    seg_crop = slice(-(pf//2 + 1), None)
                    w_crop = slice(0, pf//2 + 1)
                weighted_segment = (segments[seg_crop] == n_i) * plane_weights[w_crop]
                comp[j] = np.max(weighted_segment, axis=0)
                start = seg_crop.stop
            S[:, i] = comp.reshape(np.prod(ts[1:]))

        # format search neighborhoods
        B = neighborhood_by_weights(
            S, segments_shape=ts[1:],
            sigma=neighborhood_sigma,
            spacing=time_series_spacing,
        )

        # format the temporal components
        H = np.zeros((n_labels, ts[0]), dtype=np.float32)

        # run nmf
        if use_gpu:
            S_res, H_res, objectives = nmf_pytorch(
                V, S, H, B,
                **kwargs,
            )
        else:
            S_res, H_res, objectives = nmf(
                V, S, H, B,
                estimate_noise_component=False,
                **kwargs,
            )

        # write reconstruction
        if reconstruction_path is not None:

            # construct weighted reconstruction
            reconstruction = (S_res @ H_res).T.reshape(ts)
            reconstruction_weights = reconstruction_weights_zarr[time_series_crop]
            reconstruction *= reconstruction_weights[None, ...]
            crop_string = '_'.join([f'{x.start}x{x.stop}' for x in time_series_crop])

            # ensure correct dtype then save
            given_dtype = time_series_zarr.dtype
            if reconstruction.dtype != given_dtype:
                if given_dtype == int or np.issubdtype(given_dtype, np.integer):
                    reconstruction = np.round(reconstruction)
                reconstruction = reconstruction.astype(given_dtype)
            np.save(temp_dir_path + '/chunk_' + crop_string + '.npy', reconstruction)

        # time series for the complete cells are what we really need
        return H_res[:n_segments]


    def reconstruct_per_block(crop):

        # load crop
        crop_string = '_'.join([f'{x.start}x{x.stop}' for x in crop])
        data_path = temp_dir_path + '/chunk_' + crop_string + '.npy'
        data = np.load(data_path)
 
        # update the zarr, remove the file
        current = reconstruction_zarr[(slice(None),) + crop]
        reconstruction_zarr[(slice(None),) + crop] = current + data
        os.remove(data_path)
        return True


    @cluster_decorator
    def map_all_blocks(cluster=None, cluster_kwargs={}):

        # submit all blocks
        futures = cluster.client.map(
            nmf_per_block,
            segment_unions,
            time_series_crops,
            segments_crops,
        )
        time_series = cluster.client.gather(futures)

        # run reconstruction loop
        if reconstruction_path is not None:

            # loop until all blocks are written
            block_shape = reconstruction_zarr.chunks[1:]
            written = np.zeros(len(segment_unions), dtype=bool)
            while not np.all(written):

                # determine which compute blocks can write in parallel
                write_batch = []
                block_flags = np.ones(reconstruction_zarr.cdata_shape[1:], dtype=bool)
                for iii, crop in enumerate(time_series_crops):
                    if written[iii]: continue
                    f = lambda x, y: slice(x.start//y, (x.stop//y)+1)
                    blocks_touched = tuple(f(x, y) for x, y in zip(crop, block_shape))
                    if np.all(block_flags[blocks_touched]):
                        write_batch.append(crop)
                        written[iii] = True
                        block_flags[blocks_touched] = False

                # submit write batch
                print(f'WRITING {len(write_batch)} BLOCKS')
                futures = cluster.client.map(
                    reconstruct_per_block, write_batch,
                )
                batch_written = cluster.client.gather(futures)

        # return all time series
        return time_series

    # execute everything
    print('LAUNCHING CLUSTER, MAPPING ALL BLOCKS')
    segment_time_components = map_all_blocks(
        cluster=cluster, cluster_kwargs=cluster_kwargs,
    )

    # construct the return array and send it
    output_shape = (n_output_segments, time_series_zarr.shape[0])
    all_time_components = np.empty(output_shape, dtype=np.float32)
    all_time_components.fill(np.nan)
    for segment_union, components in zip(segment_unions, segment_time_components):
        for segment_id, component in zip(segment_union, components):
            all_time_components[segment_id-1, :] = component
    return all_time_components

