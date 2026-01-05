import torch
import torch.nn as nn
import numpy as np
from torch.autograd.functional import hessian
from torch.func import functional_call
import math

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

def compute_min_eigenvalue(model, loss, max_eig_val, num_iters=50, device='cpu'):
    """Estimates Lambda Min using Spectral Shifting."""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    v = torch.randn(num_params, device=device)
    v = v / torch.norm(v)
    for _ in range(num_iters):
        Hv = hessian_vector_product(model, loss, v)
        Hv_shifted = Hv - (max_eig_val * v)
        shifted_eig_val = torch.dot(v, Hv_shifted).item()
        v = Hv_shifted / torch.norm(Hv_shifted)
    return shifted_eig_val + max_eig_val

def estimate_trace(model, loss, num_samples=50, device='cpu'):
    """
    Estimates Trace(H) using Hutchinson's Method.
    Trace = E[v^T H v] where v is Rademacher (+1/-1) vector.
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trace_est = 0.0
    
    for _ in range(num_samples):
        # Rademacher distribution (random +1 or -1)
        v = torch.randint(0, 2, (num_params,), device=device).float() * 2 - 1
        Hv = hessian_vector_product(model, loss, v)
        val = torch.dot(v, Hv).item()
        trace_est += val
        
    return trace_est / num_samples

# def compute_exact_hessian(model, loss):
#     params = [p for p in model.parameters() if p.requires_grad]
#     grads = torch.autograd.grad(loss, params, create_graph=True)
#     grads_flat = torch.cat([g.view(-1) for g in grads])
#     hessian_rows = []
#     for i, g in enumerate(grads_flat):
#         grad2 = torch.autograd.grad(g, params, retain_graph=True)
#         row = torch.cat([g2.view(-1) for g2 in grad2])
#         hessian_rows.append(row)
#     return torch.stack(hessian_rows)

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
    
    print(f"Computing Hessian over dataset... (batches: {len(dataloader)})")
    
    for batch_idx, (data, target) in enumerate(dataloader):
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
        
        # 5. Accumulate
        # Since CrossEntropy(reduction='mean') divides by batch_size,
        # we multiply by batch_size to recover the sum of gradients.
        current_sum = batch_hessian_matrix * batch_size
        
        if total_hessian is None:
            total_hessian = current_sum
        else:
            total_hessian += current_sum
            
        total_samples += batch_size
        
        # Optional: Print progress every 10 batches
        if batch_idx % 10 == 0:
            print(f"Processed batch {batch_idx}/{len(dataloader)}")

    # 6. Final Average
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


def compute_hessian_stats(model, loader, cfg):
    """
    Returns comprehensive metrics dictionary.
    """
    device = cfg.device if cfg.device != "auto" else \
             ("cuda" if torch.cuda.is_available() else "cpu")
             
    results = {}

    if cfg.analysis.use_exact_hessian:
        print("   >> [Mode: Exact] Computing full matrix...")
        H = compute_dataset_hessian(model, loader, device=device)
        eigs = torch.linalg.eigvalsh(H) # Sorted low to high
        
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
        
    else:
        print(f"   >> [Mode: Approx] Power Iteration ({cfg.analysis.num_power_iters} steps)...")
        model.eval()
        
        # 1. Max Eigenvalue
        max_eig = dataset_power_iteration(model, loader, nn.CrossEntropyLoss(), cfg.analysis.num_power_iters, device)
        
        # 2. Min Eigenvalue
        min_eig = dataset_min_eigenvalue(model, max_eig, loader, cfg.analysis.num_power_iters, device)
        
        # 3. Trace Estimation (New)
        trace = estimate_trace(model, loader, nn.CrossEntropyLoss(), num_samples=20, device=device)
        
        results['mode'] = 'approx'
        results['max_eig'] = max_eig
        results['min_eig'] = min_eig
        results['trace'] = trace
        
        # Fill unavailable metrics with NaN so CSV columns match
        results['mean_eig'] = np.nan
        results['median_eig'] = np.nan
        results['neg_eig_ratio'] = np.nan
        results['eigenvalues'] = None

    # Common Derived Metric
    # Condition Number: |Max| / |Min|
    # (We use abs() because min can be negative at saddle points)
    denom = abs(results['min_eig'])
    if denom < 1e-6: denom = 1e-6 # Avoid div by zero
    results['condition_number'] = abs(results['max_eig']) / denom
    
    return results, H