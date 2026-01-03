import torch
import torch.nn as nn
import torch.optim as optim
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from models.simple_net import SimpleNet
from utils.data_loader import get_dataset, get_loo_loader
from utils.hessian_tools import compute_hessian_stats
from utils.plotting import plot_eigenvalues

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_and_measure(cfg, dataset, exclude_idx=None):
    set_seed(cfg.training.seed)
    
    device = cfg.analysis.device if cfg.analysis.device != "auto" else \
             ("cuda" if torch.cuda.is_available() else "cpu")
             
    loader = get_loo_loader(dataset, exclude_idx, cfg.data.batch_size)
    
    # Infer dims
    # dataset[0][0] gives the image tensor
    sample_shape = dataset[0][0].shape 
    input_dim = np.prod(sample_shape)
    
    model = SimpleNet(input_dim, cfg.model.hidden_dim).to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.training.lr, momentum=cfg.training.momentum)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    model.train()
    for epoch in range(cfg.training.epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
    # Measure Hessian on a consistent batch
    # (Using full dataset loader to ensure we verify on same data points)
    full_loader = get_loo_loader(dataset, exclude_index=None, batch_size=cfg.data.batch_size)
    inputs, targets = next(iter(full_loader))
    inputs, targets = inputs.to(device), targets.to(device)
    inputs = inputs.view(inputs.size(0), -1)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    stats = compute_hessian_stats(model, loss, cfg)
    return stats

@hydra.main(version_base=None, config_path="config", config_name="exp")
def run_loo_experiment(cfg: DictConfig):
    print(f"Loading Dataset: {cfg.data.dataset_name}")
    dataset, _, _ = get_dataset(cfg)
    
    # --- 1. Baseline ---
    print("\n>>> Running Baseline (Full Dataset)...")
    base_stats = train_and_measure(cfg, dataset, exclude_idx=None)
    
    print("\nBASELINE RESULTS:")
    for k, v in base_stats.items():
        if isinstance(v, (float, int)):
            print(f"  {k}: {v:.4f}")
    
    results = []
    row = {}
        
    # Add all stats from the stats dict
    for key in ['max_eig', 'min_eig', 'trace', 'condition_number', 'mean_eig', 'median_eig', 'neg_eig_ratio']:
        if key in base_stats:
            row[key] = base_stats[key]
    
    results.append(row)
    
    if cfg.analysis.use_exact_hessian:
        print("Plotting Eigenvalue Statistics:")
        plot_eigenvalues(base_stats['eigenvalues'], f'results/{cfg.experiment_name}_eigen.png')

    if not cfg.loo.enabled:
        return

    # --- 2. LOO Loop ---
    print(f"\n>>> Starting LOO Experiment (Samples: {cfg.loo.num_samples_to_test})...")
    
    indices_to_remove = np.random.choice(len(dataset), cfg.loo.num_samples_to_test, replace=False)
    
    for i, idx in enumerate(indices_to_remove):
        print(f"[{i+1}/{cfg.loo.num_samples_to_test}] Removing Index {idx}...")
        
        loo_stats = train_and_measure(cfg, dataset, exclude_idx=int(idx))
        
        if cfg.analysis.use_exact_hessian:
            print("Plotting Eigenvalue Statistics:")
            plot_eigenvalues(loo_stats['eigenvalues'], 
                             f'results/{cfg.experiment_name}_eigen_loo_idx_{idx}.png')
            
        # Compile Row
        row = {'sample_idx': idx}
        
        # Add all stats from the stats dict
        for key in ['max_eig', 'min_eig', 'trace', 'condition_number', 'mean_eig', 'median_eig', 'neg_eig_ratio']:
            if key in loo_stats:
                row[key] = loo_stats[key]
                
                # Calculate Deltas (Metric - Baseline)
                # Note: Delta is meaningless if value is NaN
                if key in base_stats and not np.isnan(loo_stats[key]):
                    row[f'delta_{key}'] = loo_stats[key] - base_stats[key]
        
        results.append(row)

    # --- 3. Save ---
    df = pd.DataFrame(results)
    
    # Reorder columns for readability
    cols = ['sample_idx'] + [c for c in df.columns if c != 'sample_idx']
    df = df[cols]
    
    csv_path = "loo_hessian_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n>>> Experiment Complete.")
    print(f"Saved metrics to: {csv_path}")
    print(df.head())

if __name__ == "__main__":
    run_loo_experiment()