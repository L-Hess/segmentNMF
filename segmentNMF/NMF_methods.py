"""functions implementing the NMF algorithm for demixing neural signals using segmentation data"""

import numpy as np
import torch
from tqdm import tqdm

# ======================================================================================================================
# Helper functions
# ======================================================================================================================

def pearsonr_mat(X, Y, axis=0):
    """Calculates the Pearson correlation coefficient between each row or columns of two matrices X and Y,
     both of which should have the exact same shape.

    Args:
        X: A 2D NumPy array.
        Y: A 2D NumPy array with the same shape as X
        axis: An integer indicating the axis along which to calculate the Pearson correlation coefficient.

    Returns:
        A NumPy array containing the Pearson correlation coefficient between each row or column of X and Y.
    """

    # Check if the dimensions of X and Y match.
    if X.shape != Y.shape:
        raise ValueError("The dimensions of X and Y must match.")

    # Check if X and Y are 2D arrays.
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays.")

    # Check if the axis parameter is valid.
    if axis not in range(X.ndim):
        raise ValueError("The axis parameter is out of bounds.")

    # Calculate the mean of each row or column.
    X_mean = np.nanmean(X, axis=axis)
    Y_mean = np.nanmean(Y, axis=axis)

    # Calculate the squared deviations from the mean.
    X_sq_dev = np.nansum((X - X_mean)**2, axis=axis)
    Y_sq_dev = np.nansum((Y - Y_mean)**2, axis=axis)

    # Calculate the covariance between X and Y.
    cov = np.sum((X - X_mean) * (Y - Y_mean), axis=axis)

    # Calculate the Pearson correlation coefficient.
    corr = cov / (np.sqrt(X_sq_dev * Y_sq_dev) + 1e-12)

    return corr


from scipy.stats import pearsonr

def compare_with_true(X, Y, axis=0):
    """Calculate Pearson correlation coefficient for each component

    Args:
        X: A 2D NumPy array.
        Y: A 2D NumPy array with the same shape as X
        axis: An integer indicating the axis along which to calculate the Pearson correlation coefficient.
              Note this axis parameters has the opposite definition from the axis parameter for the
              pearsonr_mat function

    Returns:
        A NumPy array containing the Pearson correlation coefficient between each row or column of X and Y.
    """
    correlations = []
    for i in range(X.shape[axis]):
        correlation, _ = pearsonr(X[i], Y[i])
        correlations.append(correlation)
    return np.array(correlations)


# ======================================================================================================================
# Step-size estimation
# ======================================================================================================================

def line_search_step_size(V, S, H, gradient, objective_function, gradient_str='S', lr=1.0, alpha=0.9999, beta=0.99):
    """Estimate the step size for a single parameter using a line search.

    Args:
        V: A PyTorch tensor representing the input data matrix.
        S: A PyTorch tensor representing the spatial components matrix.
        H: A PyTorch tensor representing the temporal components matrix.
        gradient: A PyTorch tensor representing the gradient of the objective function with respect to the parameter.
        objective_function: A function that takes the input data matrix, the spatial components matrix, and the temporal components matrix and returns the objective value.
        gradient_str: A string indicating whether gradient is for S or H. Only valid values are 'S' and 'H'
        lr: The initial learning rate.
        alpha: The strong Wolfe line search parameter.
        beta: The backtracking factor.

    Returns:
        A PyTorch tensor representing the step size for the parameter.
    """

    # Calculate the initial step size
    step_size = lr

    # Calculate the objective function at the current parameters
    objective = objective_function(V, S, H)

    grad_in = gradient.reshape(-1) @ gradient.reshape(-1)
    # print(grad_in)

    # Perform the line search
    while True:
        if gradient_str == 'S':
            S_step = S + step_size * gradient
            objective_step = objective_function(V, S_step, H)
        elif gradient_str == 'H':
            H_step = H + step_size * gradient
            objective_step = objective_function(V, S, H_step)

        # # Calculate the strong Wolfe condition
        # descent_condition = objective - objective_step <= alpha * step_size * grad_in

        # If the strong Wolfe condition is satisfied, then accept the update
        if objective_step < objective*alpha:
            break

        # Otherwise, reduce the step size and try again
        else:
            step_size *= beta
            print(gradient_str + ': new line search step size {}'.format(step_size))

    print(gradient_str + ': final step size {}'.format(step_size))
    return step_size


# ======================================================================================================================
# Numpy-based implementation
# ======================================================================================================================

def frobenius_norm(V, S, H):
    """Calculate the Frobenius norm objective.

    Args:
        V: A NumPy array representing the input data matrix.
        S: A NumPy array representing the spatial components matrix.
        H: A NumPy array representing the temporal components matrix.

    Returns:
        The Frobenius norm of the residual V - S @ H; scalar value.
    """

    # Calculate the Frobenius norm objective
    residual = V - S @ H
    objective = np.linalg.norm(residual, 'fro')
    return objective


def nmf(V, S_init, H_init, B, H_true=None, num_iterations=100, update_int=10, H_lr=1,
                S_lr=1e-6, objective_threshold=1e-6, estimate_noise_component=True,
                min_iterations: int = 50):
    """Compute NMF using gradient descent

    Args:
        V: A NumPy array representing the input data matrix.
        S_init: A NumPy array representing the initial spatial components matrix.
        H_init: A NumPy array representing the initial temporal components matrix.
        B: A NumPy array representing the neighborhood matrix.
        H_true: A NumPy array representing the true temporal components matrix (optional).
        num_iterations: The number of iterations to run the algorithm for (default: 100).
        update_int: The interval at which to display progress updates (default: 10).
        H_lr: The learning rate for the H matrix (default: 1).
        S_lr: The learning rate for the S matrix (default: 1e-6).
        objective_threshold: The stopping criterion for the objective (default: 1e-6).
        estimate_noise_component: Whether to estimate a noise component (default: True).
        min_iterations: The minimum number of iterations to run the algorithm for (default: 50).

    Returns:
        A tuple of NumPy arrays representing the learned spatial components matrix (S), temporal components matrix (H), and objective values (objectives).
    """

    n_components = H_init.shape[0]

    # If specified, add an additional 'noise' component
    if estimate_noise_component:
        noise_component = np.ones_like(S_init[:, 0]) / S_init.shape[0]
        noise_time_component = np.zeros_like(H_init[0, :])
        neighborhood_component = np.ones_like(B[:, 0])
        S_init = np.concatenate((S_init, noise_component[:, None]), axis=1)
        H_init = np.concatenate((H_init, noise_time_component[None, :]), axis=0)
        B = np.concatenate((B, neighborhood_component[:, None]), axis=1)

    objectives = []
    correlations = []
    H_gradients = []
    S_gradients = []

    print('Run gradient NMF | max iterations: {}'.format(num_iterations))
    # Perform projected gradient iterations
    for i in tqdm(range(num_iterations)):

        # Update H matrix with dynamically estimated step size
        # XXX I'm unsure about relu on the gradient. The optimization should have the
        #     opportunity to correct previous over estimates by subtracting from them.
        #     Can we apply relu to H itself after adding the gradient? This seems
        #     conceptually more correct, since it is H that must be non-negative.
        H_gradient = np.maximum(0, S.T @ (V - S @ H))
        H_step_size = H_lr / np.sum(S * S, axis=0)[:, None]
        H += H_step_size * H_gradient
        H[np.isnan(H)] = 1e-12  # XXX theoretically there should never be any NaNs. Should investigate.

        # Update S matrix with spatial constraint
        # All values outside of neighborhood B are set to 1e-12 for stability
        S_gradient = np.maximum(0, (V - S[:, :n_components] @ H[:n_components]) @ H[:n_components].T)
        S_step_size = S_lr / np.sum(H[:n_components] * H[:n_components], axis=1)[None, :]
        S[:, :n_components] += S_step_size * S_gradient
        S[np.logical_not(B)] = 1e-12

        # Save gradient steps
        S_gradients.append(np.mean(S_step_size * S_gradient))
        H_gradients.append(np.mean(H_step_size * H_gradient))

        # Calculate Frobenius norm objective
        objective = frobenius_norm(V, S, H)
        objectives.append(objective)

        # Display progress update
        if i % update_int == 0:
            progress_str = 'Iteration {} | Objective: {:.6f} | avg H step {:.10f} | avg S step {:.10f}'.format(
                i, objectives[i], np.mean(H_gradients[i]), np.mean(S_gradients[i]))
            if H_true is not None:
                correlation = pearsonr_mat(H.roll(-1, axis=0)[:H_true.shape[0]], H_true, axis=0)
                progress_str += ' | avg corr {:.5f} | min corr {:.5f}'.format(np.mean(correlations[i]),
                                                                              np.min(correlations[i]))
            tqdm.tqdm.write(progress_str)

        # Check stopping criterion
        if objective_threshold is not None and i > min_iterations:
            if abs(objectives[i] - objectives[i - 1]) < objective_threshold:
                print('Reached stopping criterion (d objective < {})'.format(objective_threshold))
                break
            if objectives[i] > objectives[i - 1]:
                print('Objective increased: [i-1]: {}, [i]: {}'.format(objectives[i-1], objectives[i]))
                break

    return S, H, objectives

# ======================================================================================================================
# Pytorch-based implementation
# ======================================================================================================================

def frobenius_norm_pytorch(V, S, H):
    """Calculate the Frobenius norm objective.

    Args:
        V: A PyTorch tensor representing the input data matrix.
        spatial_components: A PyTorch tensor representing the spatial components matrix.
        temporal_components: A PyTorch tensor representing the temporal components matrix.

    Returns:
        A PyTorch tensor representing the Frobenius norm objective.
    """

    # Calculate the Frobenius norm objective
    residual = V - S @ H
    objective = torch.norm(residual, 'fro').item()
    return objective


def nmf_pytorch(V, S_init, H_init, B, H_true=None, num_iterations=100, update_int=10, H_lr: float = 1.,
                S_lr: float = 1, objective_threshold: float = 1e-2, min_iterations: int = 50):

    """Compute NMF using gradient descent

    Args:
        V: A PyTorch tensor representing the input data matrix.
        S_init: A PyTorch tensor representing the initial spatial components matrix.
        H_init: A PyTorch tensor representing the initial temporal components matrix.
        B: A PyTorch tensor representing the neighborhood matrix.
        H_true: A PyTorch tensor representing the true temporal components matrix (optional).
        num_iterations: The number of iterations to run the algorithm for (default: 100).
        update_int: The interval at which to display progress updates (default: 10).
        H_lr: The learning rate for the H matrix (default: 1).
        S_lr: The learning rate for the S matrix (default: 1e-6).
        objective_threshold: The stopping criterion for the objective (default: 1e-6).
        estimate_noise_component: Whether to estimate a noise component (default: True).
        min_iterations: The minimum number of iterations to run the algorithm for (default: 50).

    Returns:
        A tuple of PyTorch tensors representing the learned spatial components matrix (S), temporal components matrix (H), and objective values (objectives).
    """

    # Convert numpy arrays to PyTorch tensors
    V = torch.tensor(V, dtype=torch.float64)
    S_init = torch.tensor(S_init, dtype=torch.float64)
    H_init = torch.tensor(H_init, dtype=torch.float64)
    B = torch.tensor(B, dtype=torch.float64)

    n_components = H_init.shape[0]

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device: {}'.format(device))

    # Move tensors to the GPU if available
    V = V.to(device)
    S = S_init.to(device)
    H = H_init.to(device)
    B = B.to(device)

    objectives = []
    correlations = []
    H_gradients = []
    S_gradients = []

    # messing around with line search, keeping the value of the last iteration as the starting point for the next
    # With the idea that, during training, the learning rate should keep decreasing
    S_step_size = S_lr
    H_step_size = H_lr

    print('Run gradient NMF | max iterations: {}'.format(num_iterations))
    # Perform projected gradient iterations
    for i in tqdm(range(num_iterations)):

        # Update H matrix with dynamically estimated step size
        H_gradient = S.T @ (V - S @ H)
        H_step_size = (H_lr / torch.diag(S.T @ S))[:, None]
        # H_lr = line_search_step_size(V, S, H, H_gradient, gradient_str='H',
        #                                     objective_function=frobenius_norm_pytorch,
        #                                     lr=H_step_size, alpha=1, beta=0.9)
        H.add_(H_step_size * H_gradient)
        H[torch.isnan(H)] = 1e-12
        H = torch.relu(H)

        # Update S matrix with spatial constraint
        # All values outside of neighborhood B are set to 1e-12 for stability
        S_gradient = (V - S @ H) @ H.T
        S_step_size = S_lr #/ torch.diag(H[:n_components] @ H[:n_components].T)[None, :]
        # S_lr = line_search_step_size(V, S, H, S_gradient, gradient_str='S',
        #                              objective_function=frobenius_norm_pytorch,
        #                              lr=S_step_size, alpha=1.01, beta=0.1)
        S.add_(S_step_size * S_gradient)
        S[torch.logical_not(B)] = 1e-12
        S = torch.relu(S)

        # Save gradient steps
        S_gradients.append(torch.mean(S_step_size * S_gradient).item())
        H_gradients.append(torch.mean(H_step_size * H_gradient).item())

        # Calculate Frobenius norm objective
        objective = frobenius_norm_pytorch(V, S, H)
        objectives.append(objective)

        # Display progress update
        if i % update_int == 0:
            progress_str = 'Iteration {} | Objective: {:.6f} | avg H step {:.10f} | avg S step {:.10f}'.format(
                i, objectives[i], H_gradients[i], S_gradients[i])
            if H_true is not None:
                # correlations.append(pearsonr_mat(np.roll(H[:H_true.shape[0]].cpu().numpy(), 0, axis=0), H_true, axis=0))
                correlations.append(compare_with_true(H.cpu().numpy(), H_true))
                progress_str += ' | avg corr {:.5f} | min corr {:.5f}'.format(np.nanmean(correlations[-1]),
                                                                              np.nanmin(correlations[-1]))
            print(progress_str)

        # Check stopping criterion
        if objective_threshold is not None \
                and i > min_iterations \
                and abs(objectives[i] - objectives[i - 1]) < objective_threshold or objectives[i] > objectives[i - 1]:
            print('Reached stopping criterion (d objective < {})'.format(objective_threshold))
            break

    # Move tensors back to the CPU
    S = S.cpu().numpy()
    H = H.cpu().numpy()

    if H_true is not None:
        return S, H, objectives, correlations
    else:
        return S, H, objectives
