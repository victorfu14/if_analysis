import torch
import torch.nn as nn
import torch.optim as optim
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from models.nnet import SimpleNet, MNISTNet
from utils.data_loader import get_dataset, get_loo_loader
from utils.hessian_tools import compute_hessian_stats, compute_influence_vs_validation
from utils.plotting import plot_eigenvalues
from torch.utils.data import Subset


MODEL_FUNCTION = {
    'mnist': MNISTNet,
    'simple': SimpleNet
}

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_and_measure(cfg, model, train_set, exclude_idx=None):
    set_seed(cfg.training.seed)
    
    device = cfg.device if cfg.device != "auto" else \
             ("cuda" if torch.cuda.is_available() else "cpu")
             
    loader = get_loo_loader(train_set, exclude_idx, cfg.data.batch_size)
    
    optimizer = optim.SGD(model.parameters(), lr=cfg.training.lr, momentum=cfg.training.momentum)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    model.train()
    for epoch in range(cfg.training.epochs):
        l = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            # data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            l += loss.item() * target.size(0)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}, loss: {l / len(train_set)}')
    
    # eval_loader = get_loo_loader(train_set, exclude_index=None, batch_size=cfg.data.batch_size)

    stats, hessian = compute_hessian_stats(model, loader, cfg)
    return model, stats, hessian

@hydra.main(version_base=None, config_path="config", config_name="exp")
def run_loo_experiment(cfg: DictConfig):
    print(f"Loading Dataset: {cfg.data.dataset_name}")
    device = cfg.device if cfg.device != "auto" else "cuda"
    dataset, input_dim, _ = get_dataset(cfg)
    
    model = SimpleNet(input_dim, cfg.model.hidden_dim).to(device)
    
    # --- 1. Baseline ---
    print("\n>>> Running Baseline (Full Dataset)...")
    _, base_stats, _ = train_and_measure(cfg, model, dataset, exclude_idx=None)
    
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
        model = SimpleNet(input_dim, cfg.model.hidden_dim).to(device)
        
        _, loo_stats, _ = train_and_measure(cfg, model, dataset, exclude_idx=int(idx))
        
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


@hydra.main(version_base=None, config_path="config", config_name="exp")
def run_influence_analysis(cfg: DictConfig):
    set_seed(cfg.training.seed)
    device = cfg.device if cfg.device != "auto" else "cuda"
    print(f"Using device: {device}")
    
    # 1. Load Data
    print(f"Loading Dataset: {cfg.data.dataset_name}")
    dataset, input_dim, _ = get_dataset(cfg) 
    train_size = int(0.6 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Loaders for compute IF (subsampled)
    num_samples = 100
    indices = torch.randperm(len(train_set))[:num_samples]
    if_set = Subset(train_set, indices)
    if_loader = DataLoader(if_set, batch_size=1, shuffle=False)
    
    # 2. Train Model
    Net = MODEL_FUNCTION[cfg.model.type]
    model = Net(input_dim, cfg.model.hidden_dim).to(device)
    print("\n>>> Training Model...")
    
    model, base_stats, hessian = train_and_measure(
        cfg, model, train_set
    )
    
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

    # 3. Compute Inverse Hessian (H^-1)
    print("\n>>> Computing Inverse Hessian...")
    # Grab all training data as one batch for Hessian
    H_inv = torch.linalg.inv(hessian + cfg.analysis.reg * torch.eye(hessian.shape[0], device=device))
    
    # 4. Pick a Validation Sample to Analyze
    # Let's pick the first one in the validation set
    val_sample = val_set[0]
    val_x, val_y = val_sample
    
    print(f"\n>>> Analyzing Influence on Validation Sample (True Class: {val_y})")
    
    # 5. Compute Influence for EVERY training point
    influences, _ = compute_influence_vs_validation(
        model, if_loader, val_sample, nn.CrossEntropyLoss(), H_inv, device
    )
    
    # 6. Analysis & Visualization
    df = pd.DataFrame(influences)
    df.to_csv(f"results/if_{cfg.experiment_name}.csv", index=False)
    
    # Sort by Influence
    # Positive Score = Harmful (Gradient aligns with Error)
    # Negative Score = Helpful (Gradient opposes Error)
    df_sorted = df.sort_values(by="influence_score", ascending=False)
    
    # --- 2. LOO Loop ---
    print(f"\n>>> Starting LOO Experiment (Samples: {cfg.loo.num_samples_to_test})...")
    
    indices_to_remove = np.random.choice(len(train_set), cfg.loo.num_samples_to_test, replace=False)
    
    for i, idx in enumerate(indices_to_remove):
        print(f"[{i+1}/{cfg.loo.num_samples_to_test}] Removing Index {idx}...")
        model = Net(input_dim, cfg.model.hidden_dim).to(device)
        
        model, loo_stats, hessian = train_and_measure(cfg, model, train_set, exclude_idx=int(idx))
        
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
        
        H_inv = torch.linalg.inv(hessian + cfg.analysis.reg * torch.eye(hessian.shape[0], device=device))
        influences, _ = compute_influence_vs_validation(
            model, if_loader, val_sample, nn.CrossEntropyLoss(), H_inv, device
        )
        
        # 6. Analysis & Visualization
        df = pd.DataFrame(influences)
        df.to_csv(f"results/if_{cfg.experiment_name}_loo_idx_{idx}.csv", index=False)
        
    
    df_eigen = pd.DataFrame(results)
    
    # Reorder columns for readability
    cols = ['sample_idx'] + [c for c in df_eigen.columns if c != 'sample_idx']
    df_eigen = df_eigen[cols]
    
    csv_path = f"results/{cfg.experiment_name}_loo_hessian_metrics.csv"
    df_eigen.to_csv(csv_path, index=False)
    print(f"\n>>> Experiment Complete.")
    print(f"Saved metrics to: {csv_path}")
    print(df_eigen.head())


if __name__ == "__main__":
    run_influence_analysis()