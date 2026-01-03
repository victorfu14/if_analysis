import torch
import numpy as np

# --- Core Helper Functions ---

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

# --- Metric Calculators ---

def power_iteration(model, loss, num_iters=50, device='cpu'):
    """Estimates Lambda Max."""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    v = torch.randn(num_params, device=device)
    v = v / torch.norm(v)
    for _ in range(num_iters):
        Hv = hessian_vector_product(model, loss, v)
        eig_val = torch.dot(v, Hv).item()
        v = Hv / torch.norm(Hv)
    return eig_val

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

def compute_exact_hessian(model, loss):
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grads_flat = torch.cat([g.view(-1) for g in grads])
    hessian_rows = []
    # print(f"Computing Exact Hessian for {len(grads_flat)} params...")
    for i, g in enumerate(grads_flat):
        grad2 = torch.autograd.grad(g, params, retain_graph=True)
        row = torch.cat([g2.view(-1) for g2 in grad2])
        hessian_rows.append(row)
    return torch.stack(hessian_rows)

# --- Unified Interface ---

def compute_hessian_stats(model, loss, cfg):
    """
    Returns comprehensive metrics dictionary.
    """
    device = cfg.analysis.device if cfg.analysis.device != "auto" else \
             ("cuda" if torch.cuda.is_available() else "cpu")
             
    results = {}

    if cfg.analysis.use_exact_hessian:
        print("   >> [Mode: Exact] Computing full matrix...")
        H = compute_exact_hessian(model, loss)
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
        
        # 1. Max Eigenvalue
        max_eig = power_iteration(model, loss, cfg.analysis.num_power_iters, device)
        
        # 2. Min Eigenvalue
        min_eig = compute_min_eigenvalue(model, loss, max_eig, cfg.analysis.num_power_iters, device)
        
        # 3. Trace Estimation (New)
        trace = estimate_trace(model, loss, num_samples=20, device=device)
        
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
    
    return results