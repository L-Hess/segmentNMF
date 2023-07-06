import numpy as np
import torch
from tqdm.notebook import tqdm

# ===================================================================================================================================
# Helper functions
# ===================================================================================================================================

def compare_with_true(H, H_true):
    # Calculate Pearson correlation coefficient for each component
    correlations = []
    for i in range(H.shape[0]):
        correlation, _ = pearsonr(H[i], H_true[i])
        correlations.append(correlation)
    return correlations

# ===================================================================================================================================
# Numpy-based
# ===================================================================================================================================

from scipy.stats import pearsonr

def calculate_frobenius_norm(V, S, H):
    # Calculate the Frobenius norm objective
    objective = np.linalg.norm(V - S@H, 'fro')
    return objective

def calculate_step_size(V, S, H, gradient, direction, armijo_factor=0.5, init_step_size=1.0, threshold=1e-4):
    step_size = init_step_size
    while True:
        new_S = S - step_size * direction[0]
        new_H = H - step_size * direction[1]
        
        new_S = np.maximum(new_S, 0)
        new_H = np.maximum(new_H, 0)
        
        if calculate_frobenius_norm(V, new_S, new_H) <= calculate_frobenius_norm(V, S, H) - armijo_factor * step_size * np.linalg.norm(gradient, 'fro'):
            break
        
        step_size *= armijo_factor
        
        if step_size < threshold:
            break
    
    return step_size

def nmf_projected_gradient(V, S_init, H_init, B, H_true=None, num_iterations=100, update_int=10,
                           armijo_factor=0.5, armijo_init_step_size=1.0, armijo_threshold=1e-4,
                           objective_threshold=1e-6):
    S = S_init.copy()
    H = H_init.copy()
    
    objectives = []
    correlations = []
    
    print('-'*80)
    print('Run gradient NMF | iterations: {}'.format(num_iterations))
    print('-'*80)
    
    # Perform projected gradient iterations
    for i in range(num_iterations):
        # Update H matrix with dynamically estimated step size
        H_gradient = np.maximum(np.dot(S.T, (V - np.dot(S, H))), 0)
        H_step_size = calculate_step_size(V, S, H, H_gradient, (np.zeros_like(S), H_gradient), armijo_factor, armijo_init_step_size, armijo_threshold)
        H += H_step_size * H_gradient
        H *= 1 / np.mean(H, axis=1)[:, None]  # Normalize H by its mean
        H[np.isnan(H)] = 0
        
        # Update S matrix with spatial constraint and dynamically estimated step size
        S_gradient = np.maximum(np.dot((V - np.dot(S, H)), H.T), 0)
        S_step_size = calculate_step_size(V, S, H, S_gradient, (B * S_gradient, np.zeros_like(H)), armijo_factor, armijo_init_step_size, armijo_threshold)
        S += S_step_size * S_gradient
        S[~B] = 1e-12  # Set all values outside of neighborhood B close to zero (1e-12 for stability)
        
        # Calculate Frobenius norm objective
        objective = calculate_frobenius_norm(V, S, H)
        objectives.append(objective)
        
        # Compare H matrix with true H if provided
        if H_true is not None:
            correlation = compare_with_true(H, H_true)
            correlations.append(correlation)
            
            if i % update_int == 0:
                print('Gradient Descent | Iteration {} | Objective: {} | avg corr {:.5f} | min corr {:.5f}'.format(i, objectives[i], np.mean(correlations[i]), np.min(correlations[i])))
                
        else:
            if i % update_int == 0:
                print('Gradient Descent | Iteration {} | Objective: {}'.format(i,
                                                                               objectives[i]))
        
        # Check stopping criterion
        if objective_threshold is not None and i > 100 and abs(objectives[i] - objectives[i-1]) < objective_threshold:
            print('Reached stopping criterion (d objective < {})'.format(objective_threshold))
            break
    
    if H_true is not None:
        return S, H, objectives, correlations
    else:
        return S, H, objectives

# ===================================================================================================================================
# Pytorch-based
# ===================================================================================================================================

def calculate_frobenius_norm_pt(V, S, H):
    # Calculate the Frobenius norm objective
    residual = V - S @ H
    objective = torch.norm(residual, 'fro').item()
    return objective

def nmf_projected_gradient_pt(V, S_init, H_init, B, H_true=None, num_iterations=100, update_int=10, H_lr = 1,
                                    S_lr=1e-6, objective_threshold=1e-6, estimate_noise_component=True):
    
    # Convert numpy arrays to PyTorch tensors
    V = torch.tensor(V, dtype=torch.float64)
    S_init = torch.tensor(S_init, dtype=torch.float64)
    H_init = torch.tensor(H_init, dtype=torch.float64)
    B = torch.tensor(B, dtype=torch.float64)
    
    n_components = H_init.shape[0]
    if estimate_noise_component:
        noise_component = torch.ones_like(S_init[:, 0], dtype=torch.float64) / S_init.shape[0]
        S_init = torch.cat((S_init, noise_component[:, None]), dim=1)
        H_init = torch.cat((H_init, torch.zeros_like(H_init[0, :])[None, :]), dim=0)
        B = torch.cat((B, torch.ones_like(B[:, 0])[:, None]), dim=1)
    
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
    
    print('-'*80)
    print('Run gradient NMF | iterations: {}'.format(num_iterations))
    print('-'*80)
    
    # Perform projected gradient iterations
    for i in tqdm(range(num_iterations)):
        
#         # Update H matrix with dynamically estimated step size
#         H_gradient = torch.relu(S.T @ (V - S @ H))
#         H_step_size = (1 / torch.diag(S.T @ S))[:, None]
        
        H_gradient = torch.relu(S.T @ (V - S @ H))
        H_step_size = H_lr / torch.diag(S.T @ S)[:, None]
        H.add_(H_step_size * H_gradient)
        # H *= 1 / H.mean(dim=1, keepdim=True)  # Normalize H by its mean
        H[torch.isnan(H)] = 1e-12
        
        # Update S matrix with spatial constraint and dynamically estimated step size
        S_gradient = torch.relu((V - S[:, :n_components] @ H[:n_components]) @ H[:n_components].T)
#         S_step_size = torch.ones_like(S_gradient) / torch.diag(H @ H.T)[None, :]
#         S_step_size = S_lr / torch.diag(H[:n_components] @ H[:n_components].T)[None, :]
        S_step_size = S_lr
        S[:, :n_components].add_(S_step_size * S_gradient)
        S[torch.logical_not(B)] = 1e-12  # Set all values outside of neighborhood B close to zero (1e-12 for stability)
        # S /= (torch.sum(S, dim=0, keepdims=True) + 1e-12)
        
        # Save gradient steps
        S_gradients.append(torch.mean(S_step_size * S_gradient).item())
        H_gradients.append(torch.mean(H_step_size * H_gradient).item())
        
        # Calculate Frobenius norm objective
        objective = calculate_frobenius_norm_pt(V, S, H)
        objectives.append(objective)
        
        # Compare H matrix with true H if provided
        if H_true is not None:
            correlation = compare_with_true(H.roll(-1, dims=0)[:H_true.shape[0]].cpu().numpy(), H_true)
            correlations.append(correlation)
            
            if i % update_int == 0:
                print('Gradient Descent | Iteration {} | Objective: {:.6f} | avg H step {:.10f} | avg S step {:.10f} | \n'
                      'avg corr {:.5f} | min corr {:.5f}'.format(i, objectives[i], torch.mean(H_gradients[i]),
                                                                  torch.mean(S_gradients[i]), torch.mean(correlations[i]),
                                                                  torch.min(correlations[i])))
                
        else:
            if i % update_int == 0:
                print('Gradient Descent | Iteration {} | Objective: {:.6f} |'
                      'avg H step {:.10f} | avg S step {:.10f}'.format(i, objectives[i],
                                                                       H_gradients[i],
                                                                       S_gradients[i]))
            
        # Check stopping criterion
        if objective_threshold is not None and i > 100 and abs(objectives[i] - objectives[i-1]) < objective_threshold or objectives[i] > objectives[i - 1]:
            print('-'*80)
            print('Reached stopping criterion (d objective < {})'.format(objective_threshold))
            print('-'*80)
            break
            
    # Move tensors back to the CPU
    S = S.cpu().numpy()
    H = H.cpu().numpy()
    
    if H_true is not None:
        return S, H, objectives, correlations
    else:
        return S, H, objectives
 
