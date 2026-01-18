import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.autograd.functional import hessian


def compute_exact_hessian(model, dataloader, criterion, device='cpu'):
    """
    Computes the exact Hessian of the loss function w.r.t model parameters 
    over the entire dataset provided by the dataloader.
    
    Args:
        model: A PyTorch model (nn.Module).
        dataloader: A PyTorch DataLoader yielding (inputs, targets).
        criterion: Loss function (e.g., nn.CrossEntropyLoss). 
                   WARNING: Ensure reduction='mean' (default) or 'sum' matches logic.
        device: Device to run computations on.
        
    Returns:
        A 2D Tensor (matrix) of shape (num_params, num_params) representing 
        the Hessian.
    """
    model.to(device)
    model.eval()  # Ensure evaluation mode (disable dropout, etc.)

    # 1. Capture the original parameters to restore later
    orig_params = parameters_to_vector(model.parameters()).detach().clone()
    num_params = orig_params.numel()
    
    print(f"Computing Hessian for {num_params} parameters...")
    print(f"Hessian Matrix Shape: ({num_params}, {num_params})")
    print(f"Memory required for Hessian: ~{(num_params**2 * 4) / 1024**3:.2f} GB (float32)")

    # 2. Initialize the total Hessian accumulator
    total_hessian = torch.zeros(num_params, num_params, device=device)
    total_samples = 0

    # 3. Define a wrapper function for the Hessian calculation
    # This function takes a flat parameter vector, loads it into the model, 
    # and calculates loss.
    def loss_from_params(params_vec, batch_inputs, batch_targets):
        # Load the flat parameters into the model
        # Note: This modifies model in-place, but we restore it later
        vector_to_parameters(params_vec, model.parameters())
        
        preds = model(batch_inputs)
        return criterion(preds, batch_targets)

    # 4. Iterate through batches
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        total_samples += batch_size

        # Compute Hessian for this batch
        # We pass the current flattened params as the input to be differentiated
        current_params_vec = parameters_to_vector(model.parameters())
        
        # functional.hessian computes the Hessian of the function output w.r.t. inputs
        batch_hessian = hessian(
            lambda p: loss_from_params(p, inputs, targets),
            current_params_vec
        )
        
        # 5. Accumulate
        # If the loss is an average (mean), we must un-average it by multiplying by batch_size
        # to sum it up, then divide by total_samples at the end.
        if getattr(criterion, 'reduction', 'mean') == 'mean':
            total_hessian += batch_hessian * batch_size
        else:
            # If reduction is 'sum', we just add it directly
            total_hessian += batch_hessian
            
        # Optional: Free memory
        del batch_hessian
        torch.cuda.empty_cache()

    # 6. Normalize if the loss function was a mean
    if getattr(criterion, 'reduction', 'mean') == 'mean':
        total_hessian /= total_samples

    # 7. Restore original model parameters
    vector_to_parameters(orig_params, model.parameters())

    return total_hessian

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