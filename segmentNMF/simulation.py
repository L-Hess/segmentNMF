import numpy as np
from tqdm.notebook import tqdm

def generate_artificial_NMF_data(dims: int = 2, n_cells = [6] * 2, cell_radius: int = 5, intercell_distance: int = 1, T: int = 1000,                                          decay_step: float = .9, decay_range: int = 10, intrinsic_noise: bool = True, random_noise: bool = True):

    pad = decay_range*2

    # ====================================================================================================================
    # Generate cell centers
    # ====================================================================================================================
    
    print('Generating cell centers')
    
    # Get total view size
    view_size = (cell_radius*2 + intercell_distance/2) * np.array(n_cells) + intercell_distance*2 + pad*2
    view_size = view_size.astype('int')

    # Create cell center coordinates
    cell_centers = np.meshgrid(*[np.arange(intercell_distance + cell_radius + pad, 
                                           view_size[d_i] - (intercell_distance + cell_radius),
                                           cell_radius*2+intercell_distance/2) for d_i in range(dims)])
    cell_centers = np.stack([cell_centers[d_i].flatten() for d_i in range(dims)]).astype('int')

    # ====================================================================================================================
    # Generate masks
    # ====================================================================================================================

    print('Generating cell masks')
    
    # Generate a basic mask for a cell
    center = [cell_radius]*dims
    mask_coords = np.meshgrid(*[range(cell_radius*2) for _ in range(dims)])
    mask_coords = np.stack([mask_coords[d_i].flatten() for d_i in range(dims)])

    # Get mask indices
    cell_inds = np.sum((mask_coords.T - center)**2, axis=1) < cell_radius**2
    mask_inds = mask_coords.T[cell_inds]

    masks = np.zeros(view_size)
    for m_i, ctr in enumerate(cell_centers.T):
        inds = ctr[None,:] + mask_inds - cell_radius
        for ind in inds:
            sl = tuple([ind[j] for j in range(dims)])
            masks[sl] = m_i + 1

    # ====================================================================================================================
    # Generate activity traces for each mask
    # ====================================================================================================================

    print('Generating activity traces for all cells')
    
    N_cells = cell_centers.shape[1]

    def ca_kernel(tau: int):
        return np.exp(-1/tau)

    k_desc = np.stack([ca_kernel(tau=np.random.uniform(25, 25)) for _ in range(N_cells)])
    k_rise = np.stack([ca_kernel(tau=np.random.uniform(0.5, 5)) for _ in range(N_cells)])
    a = np.stack([np.random.uniform(0.5, 1) for _ in range(N_cells)])
    t_d = np.stack([np.random.randint(0.1, 5) for _ in range(N_cells)])

    stim = np.random.poisson(lam=.05, size=(N_cells, T))

    F = np.zeros(shape=(N_cells, T+10))
    for t in range(1, T):
        F[:, t] = F[:, t-1] * k_desc + F[:, t+5] * k_rise + stim[:, t] * a

    F = F[:, :T]

    # ====================================================================================================================
    # Create volume functional activity
    # ====================================================================================================================

    print('Creating functional volume')
    
    dff = np.zeros((N_cells, *masks.shape, T))
    for m_i, ctr in tqdm(enumerate(cell_centers.T)):
        for i in range(decay_range):

            mask_coords = np.meshgrid(*[range((cell_radius + i)*2) for _ in range(dims)])
            mask_coords = np.stack([mask_coords[d_i].flatten() for d_i in range(dims)])

            # Get mask indices
            if i == 0:
                cell_inds = np.sum((mask_coords.T - center-i)**2, axis=1) < cell_radius**2
            else:
                cell_inds = (np.sum((mask_coords.T - center-i)**2, axis=1) < (cell_radius+i)**2) != \
                            (np.sum((mask_coords.T - center-i)**2, axis=1) < (cell_radius+i-1)**2)

            mask_inds = mask_coords.T[cell_inds]
            inds = ctr[None,:] + mask_inds - (cell_radius)

            d = decay_step**i
            for ind in inds:
                sl = tuple([m_i-1] + [ind[j]-i for j in range(dims)])
                dff[sl] = F[m_i-1] * d
                

    dff = np.mean(dff, axis=0)

    if random_noise:
        dff += np.random.normal(0, .01, size=(dff.shape))
        
    return cell_centers, center, masks, F, dff, intrinsic_noise