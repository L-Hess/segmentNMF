import numpy as np
import zarr
from tqdm import tqdm
import time
import pickle

from segmentNMF.file_handling import file_handler
from segmentNMF.neighborhood import neighborhood_by_dilation
from segmentNMF.neighborhood import neighborhood_by_distance

from ClusterWrap.decorator import cluster

from segmentNMF.NMF_methods import nmf_pytorch


@cluster
def distributed_volume_NMF(segments_path: str, timeseries_path: str, spacing, blocksize,
                           block_results_path: str, NMF_kwargs: dir = {},
                           segments_highres_path: str = None, spacing_highres=None,
                           max_dist: float = 5., overlap: float = 0.5, cluster=None,
                           N_max: int = 203396, estimate_optimal_order=True,
                           cluster_kwargs: dir = {}):
    """
    Perform Non-negative Matrix Factorization (NMF) in a distributed manner on large 3D volumes.

    Parameters:
        segments_path (str): Path to the 3D volume containing segments/masks.
        timeseries_path (str): Path to the 4D volume containing timeseries data.
        spacing (tuple): Voxel size in (z, x, y) dimensions for the input data.
        blocksize (tuple): Size of blocks used for computation in (z, x, y) dimensions.
        block_results_path (str)
        NMF_kwargs (dict, optional): Additional keyword arguments for NMF.
        segments_highres_path (str, optional): Path to the high-resolution 3D volume containing segments/masks.
        spacing_highres (tuple, optional): Voxel size in (z, x, y) dimensions for the high-resolution data.
        max_dist (float, optional): Maximum distance for neighborhood computation in NMF.
        overlap (float, optional): Overlap fraction between blocks.
        cluster (object, optional): Cluster object for parallel computation.
        N_max (int, optional): Maximum number of segments for results storage.
        estimate_optimal_order (bool, optional): Whether to estimate the optimal order of block computation.
        cluster_kwargs (dict, optional): Additional keyword arguments for the cluster (ClusterWrap).

    Returns:
        dict, np.ndarray, np.ndarray: A dictionary with block indices as keys and results as values,
        and two numpy arrays containing the temporal and spatial components of the NMF results.
    """

    spacing = np.array(spacing)
    if spacing_highres is not None:
        spacing_highres = np.array(spacing_highres)

    # ==================================================================================================================
    # Load data
    # ==================================================================================================================

    # Load segment and timeseries volumes
    segments_vol = file_handler(segments_path)
    timeseries_vol = file_handler(timeseries_path)

    # For future high-res masks -> spatial components
    if segments_highres_path is not None:
        segments_hr_vol = file_handler(segments_highres_path)
    else:
        segments_hr_vol = None

    # ==================================================================================================================
    # Data preprocessing
    # ==================================================================================================================

    T, lz, lx, ly = timeseries_vol.shape
    # lz, lx, ly, T = timeseries_vol.shape

    # ==================================================================================================================
    # Compute the number of blocks and set up output file
    # ==================================================================================================================

    # Get overlap and number of blocks
    blocksize = np.array(blocksize)
    overlap = np.ceil(blocksize * overlap).astype(int)
    nblocks = np.ceil(segments_vol.shape / blocksize).astype(np.int16)
    
    # Create list of block coordinates
    block_coords = []
    for (i, j, k) in np.ndindex(*nblocks):

        # Get coords of full block
        full_start = blocksize * (i, j, k) - overlap
        full_stop = full_start + blocksize + 2 * overlap
        full_start = np.maximum(0, full_start)
        full_stop = np.minimum(segments_vol.shape, full_stop)
        full_coords = tuple(slice(x, y) for x, y in zip(full_start, full_stop))

        # Get coords of inner block
        inner_start = blocksize * (i, j, k)
        inner_stop = inner_start + blocksize
        inner_start = np.maximum(0, inner_start)
        inner_stop = np.minimum(segments_vol.shape, inner_stop)
        inner_coords = tuple(slice(x, y) for x, y in zip(inner_start, inner_stop))

        block_coords.append(((i, j, k), full_coords, inner_coords))

    # ==================================================================================================================
    # Estimate optimal order of blocks if specified (based on their number of segments)
    # ==================================================================================================================

    if estimate_optimal_order:
        def optimize_block(coords):
            # fetch block slices and read segments and timeseries data
            block_index, coords, coords_inner = coords
            segments = segments_vol[coords]
            num_segments = len(np.unique(segments)[1:])

            if num_segments > 0:

                # cut off as much of the full volume as possible
                nonzero_inds = np.where(segments != 0)

                nz_min = np.min(nonzero_inds, axis=1)
                nz_max = np.max(nonzero_inds, axis=1)

                nonzero_coords = tuple([slice(min_ + coords[i].start,
                                              max_ + coords[i].start + 1)
                                        for i, [min_, max_] in enumerate(zip(nz_min, nz_max))])
                nonzero_coords_inner = tuple([slice(np.maximum(coords_inner[i].start, min_ + coords[i].start),
                                                    np.minimum(coords_inner[i].stop, max_ + coords[i].start))
                                              for i, [min_, max_] in enumerate(zip(nz_min, nz_max))])

                return num_segments, nonzero_coords, nonzero_coords_inner
            else:
                return num_segments, None, None

        print('Estimating optimal order')
        # submit all segmentations, reformat to dict[block_index] = (faces, boxes, box_ids)
        optim_results = cluster.client.gather(cluster.client.map(
            optimize_block, block_coords,
        ))

        num_segments = [res[0] for res in optim_results]
        block_coords = [(b[0], res[1], res[2]) for b, res in zip(block_coords, optim_results)]

        num_segments = np.array(num_segments)
        sort_ids = np.argsort(num_segments)[::-1]
        empty_block_ids = np.where(num_segments == 0)[0]
        sort_ids = sort_ids[:len(num_segments) - len(empty_block_ids)]
        block_coords = [block_coords[i] for i in sort_ids]

        print('Optimal order estimated')
        print('Detected and removed {}/{} empty blocks'.format(len(empty_block_ids), np.prod(nblocks)))
        print('Maximum number of segments in one block: {}'.format(np.max(num_segments)))

    # ==================================================================================================================
    # Apply NMF to each block
    # ==================================================================================================================

    def compute_block_NMF(coords):

        t_start = time.time()

        # fetch block slices and read segments and timeseries data
        block_index, coords, coords_inner = coords
        
        segments = segments_vol[coords]
        segments_inner = segments_vol[coords_inner]

        coords_ts = (slice(0, T),) + coords
        timeseries = timeseries_vol[coords_ts]
        timeseries = timeseries.astype('float32')

        # if not done already, cut off as much of the full segment volume as possible
        # TODO: check for boundaries to encompass neighborhoods
        if not estimate_optimal_order:
            nonzero_inds = np.where(segments != 0)
    
            nz_min = np.min(nonzero_inds, axis=1)
            nz_max = np.max(nonzero_inds, axis=1)
    
            nonzero_sl_masks = tuple([slice(min_, max_ + 1) for min_, max_ in zip(nz_min, nz_max)])
            nonzero_sl_ts = tuple([slice(0, T)] + [slice(min_, max_ + 1) for min_, max_ in zip(nz_min, nz_max)])
            # nonzero_sl_ts = tuple([slice(min_, max_ + 1) for min_, max_ in zip(nz_min, nz_max)] + [slice(0, T)])
    
            segments = segments[nonzero_sl_masks]
            timeseries = timeseries[nonzero_sl_ts]

        if segments_hr_vol is not None:
            diff_spacing = (spacing / spacing_highres).astype(int)
            coords_hr = tuple([slice(int(c.start*spacing), int(c.stop*spacing))
                               for c, spacing in zip(coords, diff_spacing)])
            segments_hr = segments_hr_vol[coords_hr]
            nonzero_sl_masks_hr = tuple([slice(min_ * spacing, max_ * spacing + 1)
                                         for min_, max_, spacing in zip(nz_min, nz_max, diff_spacing)])
            segments_hr = segments_hr[nonzero_sl_masks_hr]

        print('total load time {:.2f}s'.format(time.time() - t_start))

        # Translate timeseries data to [enforce positivity
        timeseries -= timeseries.min()

        # Determine which segments are within the block
        unique_segments = np.unique(segments)[1:]
        N_cells = len(unique_segments)

        # Also determine which segments are within the inner block
        unique_segments_inner = np.unique(segments_inner)[1:]

        # Only run NMF if more than 1 segment is present in the block
        if N_cells == 0:
            return block_index, ([], [], [], [], [])
        else:
            # Set up V matrix
            V = timeseries.reshape(timeseries.shape[0], np.prod(timeseries.shape[1:])).T
            V /= (V.mean(1, keepdims=True))
            V[np.isnan(V)] = 1e-12

            # Set up S matrix
            S = np.zeros(shape=(np.prod(segments.shape), N_cells))
            for i, n_i in enumerate(unique_segments):
                S[:, i] = (segments == n_i).reshape(np.prod(segments.shape))

            # Define neighborhood
            if segments_hr_vol is None:
                B = neighborhood_by_distance(S, segments_shape=segments.shape, spacing=spacing, max_dist=max_dist)

            else:
                print('Using high resolution mask image to generate neighborhoods')
                # Set up S matrix
                S_hr = np.zeros(shape=(np.prod(segments_hr.shape), N_cells))
                for i, n_i in enumerate(unique_segments):
                    S_hr[:, i] = (segments_hr == n_i).reshape(np.prod(segments_hr.shape))

                B = neighborhood_by_distance(S_hr, segments_shape=segments_hr.shape, spacing=spacing_highres,
                                             max_dist=max_dist, subsample=diff_spacing.astype(int))

            # define H_init and S_init
            H_init = np.zeros(shape=(N_cells, T))
            S_init = S / (np.sum(S, axis=0, keepdims=True) + 1e-12)

            S, H, _ = nmf_pytorch(V=V,
                                  S_init=S_init,
                                  H_init=H_init,
                                  B=B,
                                  **NMF_kwargs)

            # Take slices of the spatial components only of where there are values (corresponding to the neighborhood)
            # Save the slices and the slice values
            S_sliced = np.empty(S.shape[1], dtype=object)
            S_slices = np.empty(S.shape[1], dtype=object)
            for c_i in range(S.shape[1]):
                S_i = S[:, c_i].reshape(segments.shape)
                B_i = B[:, c_i].reshape(segments.shape)
                inds = np.where(B_i)
                sl = tuple([slice(min_, max_ + 1) for min_, max_ in zip(np.min(inds, 1), np.max(inds, 1))])
                S_sliced[c_i] = S_i[sl]
                S_slices[c_i] = sl

            print('total run time {:.2f}s'.format(time.time() - t_start))

            # We save results of each block computation as backup
            # TODO: make it possible to continue a stopped NMF run from a backup point
            res = {'unique_segments': unique_segments,
                   'unique_segments_inner': unique_segments_inner,
                   'H': H, 'S_sliced': S_sliced,
                   'S_slices': S_slices}

            pickle.dump(res, open(block_results_path + '/block_{}_results.pkl'.format(block_index), 'wb'))

            return block_index, (unique_segments, unique_segments_inner, H, S_sliced, S_slices)

    # For debugging purposes, call compute_block_NMF without DASK
    compute_block_NMF(block_coords[0])

    # Submit jobs to cluster
    print('submitting jobs')
    results = cluster.client.gather(cluster.client.map(
        compute_block_NMF, block_coords,
    ))
    results = {a: b for a, b in results}

    # ==================================================================================================================
    # Merge results across blocks
    # ==================================================================================================================

    # Create summary arrays containing data of inner blocks across all blocks
    tot_inner_segs = np.sum([len(v[1]) for v in list(results.values())])
    H_tot_inner = np.zeros(shape=(tot_inner_segs, T), dtype='float64')
    S_tot_inner = np.empty(shape=(tot_inner_segs), dtype=object)
    seg_inds_inner_tot = np.zeros(shape=(tot_inner_segs))
    start_ind = 0
    for i, [seg_inds, seg_inds_inner, H, S, _] in enumerate(results.values()):
        if len(seg_inds_inner) > 0:
            inds = np.concatenate([np.where(seg_inds == i)[0] for i in seg_inds_inner])
            H_tot_inner[start_ind:start_ind + len(seg_inds_inner)] = H[inds]
            S_ = []
            for ind in inds:
                S_.append(S[ind])
            S_tot_inner[i] = S_
            seg_inds_inner_tot[start_ind:start_ind + len(seg_inds_inner)] = seg_inds_inner
            start_ind += len(seg_inds_inner)

    # Determine whether segments are found in multiple inner blocks
    u, c = np.unique(seg_inds_inner_tot, return_counts=True)

    # Create arrays to save results in
    H_tot = np.full(shape=(np.maximum(int(np.max(u)), N_max), T), fill_value=np.nan, dtype=float)
    # S_tot = np.empty(shape=(np.maximum(int(np.max(u)), N_max)), dtype=object)

    # Save all segments present in only one inner block (no need for merging)
    one_inds = u[np.where(c == 1)[0]].astype(int)
    inds = np.concatenate([np.where(seg_inds_inner_tot == ind)[0] for ind in one_inds])
    H_tot[one_inds-1] = H_tot_inner[inds]
    # S_tot[one_inds-1] = S_tot_inner[inds]

    # Merge any segments that are present in multiple blocks
    # TODO: we can make a catch during estimation of the optimal order by keeping track of which neurons
    #  are in which blocks
    # TODO: make this actually function correctly
    if np.any(c > 8):
        print('One (or more) segments are divided over more than eight blocks (this should be impossible unless'
              ' the size of a neuron exceeds the block_size)')

    mult_inds = u[np.where(c >= 2)[0]]
    tot_segs_cs = np.cumsum([len(v[1]) for v in list(results.values())])
    for ind in tqdm(mult_inds[:]):

        # Get main indices of double components and indices into S list
        inds = np.where(seg_inds_inner_tot == ind)[0]
        S_ar = np.array([[[0] + list(np.where(i - tot_segs_cs >= 0)[0] + 1)][0][-1] for i in inds])
        # S_rest = inds - tot_segs_cs[S_ar-1]

        # Get block inds
        block_inds = [block_coords[S_ar_ind][0] for S_ar_ind in S_ar]
        N_segs = len(inds)

        # Retrieve spatial and temporal components
        # Ss = np.stack([S_tot_inner[S_ar[i]][S_rest[i]] for i in range(N_segs)])
        Hs = H_tot_inner[inds]

        sigcorr = np.corrcoef(H_tot_inner[inds])[np.triu_indices(N_segs, k=1)]

        # Merge components by meaning if volume mutual information
        # and temporal component correlation is high enough
        if np.any(sigcorr < .90):
            print('Segment {} has a temporal correlation of {} '
                  'between blocks {}'.format(int(ind), sigcorr, block_inds))
        else:
            print('Merging segment {} between blocks {} with a temporal '
                  'correlation of {}'.format(int(ind), block_inds, sigcorr))
            H_tot[int(ind)-1] = np.mean(Hs, axis=0)

    # ==================================================================================================================
    # Save the results
    # ==================================================================================================================

    temporal_components_zarr = zarr.open('./temporal_components_wholebrain_MF53.zarr', mode='w',
                              shape=H_tot.shape,
                              dtype=H_tot.dtype)
    temporal_components_zarr[:] = H_tot

    # spatial_components_zarr = zarr.open('./spatial_components.zarr', mode='w',
    #                           shape=S_tot.shape,
    #                           dtype=S_tot.dtype, object_codec=Pickle())
    # spatial_components_zarr[:] = S_tot

    return results, H_tot #, S_tot


if __name__ == '__main__':

    segments_path = '/groups/bishop/bishoplab/Luuk/segmentNMF/notebooks/NMF_development/masks_sl_small.zarr'
    segments_hr_path = '/groups/bishop/bishoplab/Luuk/segmentNMF/notebooks/NMF_development/masks_hr_sl_small.zarr'
    timeseries_path = '/groups/bishop/bishoplab/Luuk/segmentNMF/notebooks/NMF_development/ts_data_sl_small.zarr'

    # segments_path = '/groups/bishop/bishoplab/Luuk/segmentNMF/notebooks/NMF_development/masks.zarr'
    # segments_hr_path = '/groups/bishop/bishoplab/Luuk/segmentNMF/notebooks/NMF_development/masks_hr.zarr'
    # # timeseries_path = '/nrs/ahrensraverfish/fleishmang/F50/function/EMLmultiFISH_f50_6dpf_10ms_51z_2Hz_23min_VisRap_20220615_112144/motion_corrected.zarr'
    # timeseries_path = '/groups/bishop/bishoplab/Luuk/projects/multifish/motion_corrected_rechunk_344_51_24_24.zarr'

    # segments_path = '/groups/bishop/bishoplab/Luuk/segmentNMF/notebooks/masks_subsample_MF53.zarr'
    # segments_hr_path = '/groups/bishop/bishoplab/Luuk/segmentNMF/notebooks/masks_MF53.zarr'
    # # timeseries_path = '/nrs/ahrensraverfish/fleishmang/F50/function/EMLmultiFISH_f50_6dpf_10ms_51z_2Hz_23min_VisRap_20220615_112144/motion_corrected.zarr'
    # timeseries_path = '/groups/bishop/bishoplab/Luuk/projects/multifish/motion_corrected_MF53_rechunk_344_51_24_24.zarr'

    block_results_path = '/groups/bishop/bishoplab/Luuk/segmentNMF/data/block_results_crop/'

    NMF_kwargs = {'num_iterations': 250,
                  'update_int': 1,
                  'H_lr': 1e-1,
                  'S_lr': 1e-7,
                  'objective_threshold': 1e-1,
                  'min_iterations': 25}

    # some dask config settings
    config = {
        'distributed.worker.memory.target': 0.9,
        'distributed.worker.memory.spill': 0.9,
        'distributed.worker.memory.pause': 0.9,
        'distributed.comm.timeouts.connect': '300s',
    }

    # set cluster arguments
    cluster_kwargs = {
        'project': 'ahrens',
        'ncpus': 1,
        'threads': 1,
        'min_workers': 1,
        'max_workers': 100,
        'queue': 'gpu_rtx',
        'job_extra': ['-gpu "num=1"'],
        'walltime': '12:00',
        'config': config,
    }

    distributed_volume_NMF(segments_path=segments_path,
                           timeseries_path=timeseries_path,
                           segments_highres_path=segments_hr_path,
                           block_results_path=block_results_path,
                           blocksize=[1, 100, 100],
                           overlap=0.25,
                           spacing=[7.0, 0.406, 0.406],
                           spacing_highres=[1.0, 0.406, 0.406],
                           max_dist=2.5,
                           NMF_kwargs=NMF_kwargs,
                           cluster_kwargs=cluster_kwargs,
                           estimate_optimal_order=True)

