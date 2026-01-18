import torch
import torch.nn as nn
import numpy as np
from torch.autograd.functional import hessian
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.func import functional_call, vmap, grad
import math
from scipy import linalg
from tqdm import tqdm

def get_gradients_flat(model, loss):
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True)
    return torch.cat([g.view(-1) for g in grads])


def hessian_vector_product(model, loss, v):
    flat_grads = get_gradients_flat(model, loss)
    grad_v_prod = torch.dot(flat_grads, v)
    params = [p for p in model.parameters() if p.requires_grad]
    hvp = torch.autograd.grad(grad_v_prod, params, retain_graph=True)
    return torch.cat([g.view(-1) for g in hvp])

def batch_hessian_vector_product(model, loss, v):
    """
    Computes H*v for a single batch using the Double Backward trick.
    """
    # 1. Get parameters
    params = [p for p in model.parameters() if p.requires_grad]
    
    # 2. Compute Gradients (First Backward)
    # create_graph=True is CRITICAL. It allows us to differentiate 
    # these gradients again in step 4.
    grads = torch.autograd.grad(loss, params, create_graph=True)
    
    # 3. Compute dot product (Gradient * v)
    # This prepares the scalar for the second backward pass.
    # We zip the flat vector 'v' with the structured gradients.
    elem_wise_products = []
    current_idx = 0
    for g in grads:
        num_el = g.numel()
        v_slice = v[current_idx : current_idx + num_el].view(g.shape)
        elem_wise_products.append(torch.sum(g * v_slice))
        current_idx += num_el
        
    grad_v_prod = sum(elem_wise_products)
    
    # 4. Compute H*v (Second Backward)
    # Gradient of (grad * v) is H * v
    hvp = torch.autograd.grad(grad_v_prod, params, retain_graph=False)
    
    # Flatten result
    return torch.cat([g.view(-1) for g in hvp])

def dataset_power_iteration(model, dataloader, criterion=nn.CrossEntropyLoss(), num_iters=50, device='cpu'):
    """Estimates Lambda Max over the full dataset."""
    model.to(device)
    model.eval()
    
    # Initialize random vector v
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    v = torch.randn(num_params, device=device)
    v = v / torch.norm(v)
    
    eigenvalue = 0.0
    total_samples = len(dataloader.dataset)
    
    for i in range(num_iters):
        total_Hv = torch.zeros_like(v)
        
        # --- Accumulate H*v over the dataset ---
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            
            # Forward
            output = model(data)
            loss = criterion(output, target)
            
            # HVP for this batch
            # Note: We scale by batch_size because reduction='mean' divides by it
            batch_Hv = batch_hessian_vector_product(model, loss, v)
            total_Hv += batch_Hv * batch_size
            
            # Cleanup to save RAM
            model.zero_grad() 
            del loss, output
            
        # Normalize by N to get the true mean Hessian vector product
        total_Hv = total_Hv / total_samples
        
        # --- Update Step ---
        # Rayleigh Quotient
        eigenvalue = torch.dot(v, total_Hv).item()
        
        # Re-normalize v
        v = total_Hv / torch.norm(total_Hv)
        
        print(f"Iter {i+1}: Lambda Max ≈ {eigenvalue:.5f}")
        
    return eigenvalue

def dataset_min_eigenvalue(model, max_eig_val, dataloader, criterion=nn.CrossEntropyLoss(), num_iters=50, device='cpu'):
    """Estimates Lambda Min using Spectral Shifting over full dataset."""
    model.to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    v = torch.randn(num_params, device=device)
    v = v / torch.norm(v)
    
    shifted_eigenvalue = 0.0
    total_samples = len(dataloader.dataset)

    print(f"Computing Min Eigenvalue (Shifting by {max_eig_val:.4f})...")

    for i in range(num_iters):
        total_Hv = torch.zeros_like(v)
        
        # --- Accumulate H*v over the dataset ---
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            
            output = model(data)
            loss = criterion(output, target)
            
            batch_Hv = batch_hessian_vector_product(model, loss, v)
            total_Hv += batch_Hv * batch_size
            
            model.zero_grad()
            del loss, output

        # Normalize by N
        total_Hv = total_Hv / total_samples
        
        # --- Spectral Shift ---
        # We want the largest eigenval of (H - lambda_max * I)
        # That is: Hv - (lambda_max * v)
        Hv_shifted = total_Hv - (max_eig_val * v)
        
        # Rayleigh Quotient for the shifted matrix
        shifted_eigenvalue = torch.dot(v, Hv_shifted).item()
        
        # Update v
        v = Hv_shifted / torch.norm(Hv_shifted)
        
        print(f"Iter {i+1}: Shifted Eig ≈ {shifted_eigenvalue:.5f}")

    # Recover the original min eigenvalue
    # The dominant eigenvalue of the shifted matrix is (lambda_min - lambda_max)
    # So, lambda_min = shifted_val + lambda_max
    return shifted_eigenvalue + max_eig_val


def dataset_estimate_trace(model, dataloader, criterion, num_samples=50, device='cpu'):
    """
    Estimates Trace(H) over the full dataset using Hutchinson's Method.
    
    Args:
        num_samples: Number of random vectors v to average over (Monte Carlo steps).
    """
    model.to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_samples_count = len(dataloader.dataset)
    
    trace_estimate_sum = 0.0
    
    print(f"Estimating Trace (Using {num_samples} Hutchinson vectors)...")

    for i in range(num_samples):
        # 1. Generate Rademacher vector v (+1 or -1)
        # This v stays fixed for the entire pass through the dataset
        v = torch.randint(0, 2, (num_params,), device=device).float() * 2 - 1
        
        # 2. Accumulate Hv over the entire dataset
        total_Hv = torch.zeros_like(v)
        
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            
            output = model(data)
            loss = criterion(output, target)
            
            # Compute H*v for this batch
            batch_Hv = batch_hessian_vector_product(model, loss, v)
            
            # Accumulate weighted by batch size
            total_Hv += batch_Hv * batch_size
            
            model.zero_grad()
            del loss, output
            
        # 3. Normalize to get the Mean Hessian Vector Product
        total_Hv = total_Hv / total_samples_count
        
        # 4. Compute quadratic form: v^T * H * v
        # Since v is +1/-1, v^T * (Hv) is the estimator for the trace
        val = torch.dot(v, total_Hv).item()
        trace_estimate_sum += val
        
        if (i+1) % 5 == 0:
            print(f"Sample {i+1}/{num_samples}: Current Trace Est ≈ {trace_estimate_sum / (i+1):.4f}")
            
    return trace_estimate_sum / num_samples


def compute_hessian_block_stitching(hessian_blocks):
    """
    Robustly flattens the output from functional.hessian into a single 2D matrix.
    Works for ANY network (MLP, CNN, etc.) by inferring shapes from diagonal blocks.
    """
    num_params = len(hessian_blocks)
    param_sizes = []

    # 1. Determine the flattened size of each parameter
    # We look at the diagonal blocks (H_ii), which represent d^2L / dPi^2.
    # These are always square matrices of total size (N_param * N_param).
    for i in range(num_params):
        diag_block = hessian_blocks[i][i]
        # Calculate N = sqrt(total elements)
        size = int(math.sqrt(diag_block.numel()))
        param_sizes.append(size)
    
    # 2. Reshape and Stitch
    rows = []
    for i in range(num_params):
        row_stitched = []
        for j in range(num_params):
            block = hessian_blocks[i][j]
            
            # Explicitly reshape the block to (Size_i, Size_j)
            # This handles the 3D cases (Weight vs Bias) correctly
            reshaped_block = block.reshape(param_sizes[i], param_sizes[j])
            
            row_stitched.append(reshaped_block)
            
        # Concatenate columns (horizontally)
        rows.append(torch.cat(row_stitched, dim=1))
    
    # Concatenate rows (vertically)
    return torch.cat(rows, dim=0)


def compute_dataset_hessian(model, dataloader, criterion=nn.CrossEntropyLoss(), device='cpu'):
    """
    Computes the exact Hessian averaged over the entire dataset.
    
    Args:
        model: The neural network
        dataloader: Torch DataLoader containing the training set
        criterion: Loss function (must have reduction='mean')
    """
    model.eval() # Ensure we are in eval mode (disable dropout, etc)
    model.to(device)
    
    total_hessian = None
    total_samples = 0
    
    # 1. Prepare parameters for functional call
    params_dict = dict(model.named_parameters())
    param_names = list(params_dict.keys())
    param_values = tuple(params_dict.values())
    total_params = sum(p.numel() for p in param_values)
    
    print(f"Computing Hessian over dataset... (batches: {len(dataloader)})")
    print(f"Total Parameters: {total_params}")
    
    for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
        data, target = data.to(device), target.to(device)
        batch_size = data.size(0)
        
        # 2. Define the stateless loss function for this specific batch
        def batch_loss_func(*params):
            # Reconstruct params dictionary
            new_params_dict = {name: p for name, p in zip(param_names, params)}
            # Stateless forward pass
            output = functional_call(model, new_params_dict, data)
            return criterion(output, target)
        
        # 3. Compute Hessian for this batch
        # Result is a tuple of tuples
        batch_hessian_blocks = hessian(batch_loss_func, param_values)
        
        # 4. Stitch blocks into a single 2D matrix
        batch_hessian_matrix = compute_hessian_block_stitching(batch_hessian_blocks)
        # print(batch_hessian_matrix)
        
        # 5. Accumulate
        # Since CrossEntropy(reduction='mean') divides by batch_size,
        # we multiply by batch_size to recover the sum of gradients.
        current_sum = batch_hessian_matrix * batch_size
        
        if total_hessian is None:
            total_hessian = current_sum
        else:
            total_hessian += current_sum
            
        total_samples += batch_size
        

    # 6. Final Average
    final_avg_hessian = total_hessian / total_samples
    
    return final_avg_hessian



def compute_dataset_gauss_newton_hessian(model, dataloader, criterion=nn.CrossEntropyLoss(), device='cpu'):
    """
    Computes the "Gauss-Newton" Hessian approximation (Empirical Fisher) 
    using the outer product of gradients averaged over the entire dataset.
    
    Formula: H ≈ (1/N) * Σ (∇L_i * ∇L_i^T)
    
    Args:
        model: The neural network
        dataloader: Torch DataLoader containing the training set
        criterion: Loss function (must have reduction='mean' or 'sum' logic handled)
    """
    model.eval()
    model.to(device)
    
    total_hessian = None
    total_samples = 0
    
    # 1. Prepare parameters
    params_dict = dict(model.named_parameters())
    param_names = list(params_dict.keys())
    param_values = tuple(params_dict.values())
    
    # Calculate total number of parameters to pre-allocate matrix size if needed
    total_params = sum(p.numel() for p in param_values)
    
    print(f"Computing Empirical Fisher (Gauss-Newton) over dataset...")
    print(f"Total Parameters: {total_params}")
    
    for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
        data, target = data.to(device), target.to(device)
        
        # 2. Define a function to compute loss for a SINGLE sample
        # We need per-sample gradients, so we must define the loss per sample.
        def compute_sample_loss(params, sample_data, sample_target):
            # functional_call expects a dict
            new_params_dict = {name: p for name, p in zip(param_names, params)}
            
            # Unsqueeze to add batch dimension of 1, because models expect (B, ...)
            # We treat the single sample as a mini-batch of size 1
            sample_out = functional_call(model, new_params_dict, sample_data.unsqueeze(0))
            
            # Squeeze output to match target shape if necessary, or pass to criterion
            # Assuming standard CrossEntropy which handles (B, C) vs (B)
            return criterion(sample_out, sample_target.unsqueeze(0))

        # 3. Use vmap + grad to compute gradients for the whole batch at once
        # grad returns a tuple of gradients (one per parameter tensor)
        # vmap vectorizes this over the batch dimension (dim 0 of input data/target)
        compute_batch_grads = vmap(grad(compute_sample_loss), in_dims=(None, 0, 0))
        
        # batch_grads_tuple contains tensors of shape (Batch_Size, *Param_Shape)
        batch_grads_tuple = compute_batch_grads(param_values, data, target)
        
        # 4. Flatten and Concatenate gradients
        # We want a matrix G of shape (Batch_Size, Total_Params)
        flattened_grads = []
        for param_grad in batch_grads_tuple:
            # Flatten all dims except the batch dim (dim 0)
            flattened_grads.append(param_grad.view(data.size(0), -1))
            
        # Concatenate along the feature dimension
        batch_grads_matrix = torch.cat(flattened_grads, dim=1) # Shape: (B, P)
        
        # 5. Compute Outer Product (Gauss-Newton approximation)
        # Sum(g_i * g_i^T) is equivalent to G^T @ G
        # Shape: (P, B) @ (B, P) -> (P, P)
        current_gn_sum = batch_grads_matrix.T @ batch_grads_matrix
        
        # 6. Accumulate
        if total_hessian is None:
            total_hessian = current_gn_sum
        else:
            total_hessian += current_gn_sum
            
        total_samples += data.size(0)

    # 7. Final Average
    final_avg_hessian = total_hessian / total_samples
    
    return final_avg_hessian


def compute_gradient(model, input_data, target, criterion):
    """
    Computes the gradient of the loss with respect to parameters for a single sample.
    Works for MLP (1D input), CNN (3D input), etc.
    """
    
    # 1. Add Batch Dimension if missing
    # If input is 1D (Features) -> Make it (1, Features)
    # If input is 3D (Channels, H, W) -> Make it (1, Channels, H, W)
    # We check input_data.dim() because Flatten(start_dim=1) crashes on 1D inputs
    if input_data.dim() == 1 or input_data.dim() == 3:
        input_data = input_data.unsqueeze(0)

    # 2. Sanity Check for Target (Must be 1D list of labels)
    # If target is scalar (0D) -> (1,)
    if target.dim() == 0:
        target = target.unsqueeze(0)
    # If target is (Batch, 1) -> (Batch,)
    elif target.dim() > 1:
        target = target.view(-1)
        
    output = model(input_data)
    loss = criterion(output, target)
    
    params = [p for p in model.parameters() if p.requires_grad]
    
    # create_graph=False saves memory since we only need 1st order grad here
    grads = torch.autograd.grad(loss, params, create_graph=False)
    
    # 3. Flatten properly
    # We flatten each grad tensor and concatenate them into one long vector
    return torch.cat([g.view(-1) for g in grads])


def compute_influence_vs_validation(model, train_loader, val_sample, criterion, H_inv, device):
    """
    Computes influence function grad(z_val)^T * H^-1 * grad(z_train).
    """
    print("   >> Computing Gradient for Validation Sample...")
    # 1. Compute grad(z_val)
    val_input, val_target = val_sample
    if not isinstance(val_target, torch.Tensor):
        val_target = torch.tensor(val_target)
    val_input, val_target = val_input.to(device), val_target.to(device)
    
    grad_val = compute_gradient(model, val_input, val_target, criterion)
    
    # 2. Compute 'Preconditioned' Gradient: s_test = H^-1 * grad(z_val)
    # This vector represents the direction we want to move parameters to help z_val
    s_test = torch.mv(H_inv, grad_val)
    
    print(f"   >> Calculating Influence for {len(train_loader.dataset)} training samples...")
    influences = []
    
    # 3. Loop over all training points
    # We set batch_size=1 in the loader for this step to get per-sample gradients
    for i, (tr_input, tr_target) in enumerate(train_loader):
        tr_input, tr_target = tr_input.to(device), tr_target.to(device)
        
        # Compute grad(z_train)
        grad_train = compute_gradient(model, tr_input, tr_target, criterion)
        
        # Influence = s_test dot grad_train
        # Formula: grad(z_val)^T H^-1 grad(z_train)
        score = torch.dot(s_test, grad_train).item()
        
        influences.append({
            'train_idx': i,
            'influence_score': score,
            'target_class': tr_target.item()
        })
        
    return influences, s_test


def robust_eigvalsh(H):
    """
    Computes eigenvalues of H. Falls back to CPU and float64 if CUDA fails.
    """
    try:
        # Try standard GPU decomposition first
        return torch.linalg.eigvalsh(H)
    except RuntimeError as e:
        # Check if it's the specific CUDA error or a generic one
        if "CUSOLVER" in str(e) or "MAGMA" in str(e):
            print(f"CUDA Solver failed ({str(e)}). Falling back to CPU float64.")
            
            # 1. Move to CPU
            H_cpu = H.detach().cpu()
            
            # 2. Convert to float64 (Double Precision)
            # This is crucial. float32 often lacks the precision for degenerate Hessians.
            H_cpu = H_cpu.to(torch.float64)
            
            # 3. Decompose on CPU
            eigs_cpu = torch.linalg.eigvalsh(H_cpu)
            
            # 4. Move result back to original device and type
            return eigs_cpu.to(H.device).to(H.dtype)
            # return eigs_cpu
        else:
            # If it's a different error, re-raise it
            raise e

# # --- Usage in your code ---
# # eigs = torch.linalg.eigvalsh(H)  <-- Delete this
# eigs = robust_eigvalsh(H)


def compute_hessian_stats(model, loader, criterion, cfg):
    """
    Returns comprehensive metrics dictionary.
    """
    device = cfg.device if cfg.device != "auto" else \
             ("cuda" if torch.cuda.is_available() else "cpu")
             
    results = {}

    if cfg.analysis.method == "exact_hessian":
        print(">> [Mode: Exact] Computing full matrix...")
        H = compute_dataset_hessian(model, loader, criterion, device=device)
    
    if cfg.analysis.method == "fisher":
        print(">> [Mode: FIM] Computing FIM estimate...")
        H = compute_dataset_gauss_newton_hessian(model, loader, criterion, device=device)  
        
        
    if torch.isnan(H).any():
        print("Matrix contains NaNs!")
    if torch.isinf(H).any():
        print("Matrix contains Infs!")
    
    eigs = robust_eigvalsh(H) # Sorted low to high
    
    # Basic Stats
    results['mode'] = 'exact'
    results['max_eig'] = eigs[-1].item()
    results['min_eig'] = eigs[0].item()
    results['trace'] = torch.sum(eigs).item()
    
    # Advanced Distribution Stats
    results['mean_eig'] = torch.mean(eigs).item()
    results['median_eig'] = torch.median(eigs).item()
    results['neg_eig_ratio'] = (torch.sum(eigs < 0).float() / len(eigs)).item()
    
    results['eigenvalues'] = eigs # For plotting
        
    
    # else:
    #     print(f"   >> [Mode: Approx] Power Iteration ({cfg.analysis.num_power_iters} steps)...")
    #     model.eval()
        
    #     # 1. Max Eigenvalue
    #     max_eig = dataset_power_iteration(model, loader, nn.CrossEntropyLoss(), cfg.analysis.num_power_iters, device)
        
    #     # 2. Min Eigenvalue
    #     min_eig = dataset_min_eigenvalue(model, max_eig, loader, cfg.analysis.num_power_iters, device)
        
    #     # 3. Trace Estimation (New)
    #     trace = estimate_trace(model, loader, nn.CrossEntropyLoss(), num_samples=20, device=device)
        
    #     results['mode'] = 'approx'
    #     results['max_eig'] = max_eig
    #     results['min_eig'] = min_eig
    #     results['trace'] = trace
        
    #     # Fill unavailable metrics with NaN so CSV columns match
    #     results['mean_eig'] = np.nan
    #     results['median_eig'] = np.nan
    #     results['neg_eig_ratio'] = np.nan
    #     results['eigenvalues'] = None

    # Common Derived Metric
    # Condition Number: |Max| / |Min|
    # (We use abs() because min can be negative at saddle points)
    denom = abs(results['min_eig'])
    if denom < 1e-6: denom = 1e-6 # Avoid div by zero
    results['condition_number'] = abs(results['max_eig']) / denom
    
    return results, H