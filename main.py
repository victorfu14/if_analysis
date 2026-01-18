import torch
import torch.nn as nn
import torch.optim as optim
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_and_measure(cfg, model, train_set, val_set, exclude_idx=None):
    set_seed(cfg.training.seed)

    device = cfg.device if cfg.device != "auto" else \
             ("cuda" if torch.cuda.is_available() else "cpu")

    loader = get_loo_loader(train_set, exclude_idx, cfg.data.batch_size)
    val_loader = get_loo_loader(val_set, None, cfg.data.batch_size)

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

        model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        
        # Disable gradient calculation for validation
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                
                # Get the index of the max log-probability (predicted class)
                _, predicted = torch.max(output.data, 1)
                
                total += target.size(0)
                correct += (predicted == target).sum().item()

        val_accuracy = 100 * correct / total

        print(f'Epoch {epoch+1}: Train Loss: {l / len(train_set):.4f} | Validation Acc: {val_accuracy:.2f}%')

    model_path = f'results/{cfg.experiment_name}/model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    # eval_loader = get_loo_loader(train_set, exclude_index=None, batch_size=cfg.data.batch_size)
    
    # put larger batchsize to speed up hessian process (as long as it fits in mem)
    new_loader =  get_loo_loader(train_set, exclude_idx, batch_size=4096)

    stats, hessian = compute_hessian_stats(model, new_loader, criterion, cfg)
    return model, stats, hessian


@hydra.main(version_base=None, config_path="config", config_name="exp_mnist")
def run_influence_analysis(cfg: DictConfig):
    os.makedirs(f'results/{cfg.experiment_name}', exist_ok=True)
    set_seed(cfg.training.seed)
    device = cfg.device if cfg.device != "auto" else "cuda"
    print(f"Using device: {device}")

    # Load Data
    print(f"Loading Dataset: {cfg.data.dataset_name}")
    dataset, input_dim, _ = get_dataset(cfg)
    train_size = int(0.6 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Loaders for compute IF (subsampled)
    num_samples = 100
    indices = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(42))[:num_samples]
    if_set = Subset(train_set, indices)
    if_loader = DataLoader(if_set, batch_size=1, shuffle=False)

    # Train Model
    Net = MODEL_FUNCTION[cfg.model.type]
    model = Net(input_dim, cfg.model.hidden_dim).to(device)
    print("\n>>> Training Model...")

    model, base_stats, hessian = train_and_measure(
        cfg, model, train_set, val_set
    )
    
    # matrix = hessian.detach().cpu().numpy()
    # np.savetxt(f'results/{cfg.experiment_name}/hessian.txt', matrix, fmt='%.4f', delimiter='\t')
    np.savetxt(f'results/{cfg.experiment_name}/eigenvalue.txt', base_stats['eigenvalues'].detach().cpu().numpy(), fmt='%.4f', delimiter='\t')

    results = []
    row = {}

    # Add all stats from the stats dict
    for key in ['max_eig', 'min_eig', 'trace', 'condition_number', 'mean_eig', 'median_eig', 'neg_eig_ratio']:
        if key in base_stats:
            row[key] = base_stats[key]

    results.append(row)
    if cfg.analysis.use_exact_hessian:
        print("Plotting Eigenvalue Statistics:")
        plot_eigenvalues(base_stats['eigenvalues'], f'results/{cfg.experiment_name}/eigen.png')

    val_sample = val_set[0]
    val_x, val_y = val_sample
    print(f"\n>>> Analyzing Influence on Validation Sample (True Class: {val_y})")
    
    regs = np.logspace(-5, -1, num=10)
    # regs = [0]
    norms = {}
    for reg in regs:
        print(f"\n>>> Computing Inverse Hessian, lambda={reg}")
        H_inv = torch.linalg.inv(hessian + reg * torch.eye(hessian.shape[0], device=device))
        # H_inv = torch.eye(hessian.shape[0], device=device)
        norm = torch.linalg.matrix_norm(H_inv, ord=2)
        norms[float(reg)] = norm.item()
        print(f'Norm of regularized inverse: {norm}')

        influences, _ = compute_influence_vs_validation(
            model, if_loader, val_sample, nn.CrossEntropyLoss(), H_inv, device
        )

        df = pd.DataFrame(influences)
        df.to_csv(f"results/{cfg.experiment_name}/if_reg_{reg}.csv", index=False)
        print(df['influence_score'].var())
        
        df_sorted = df.sort_values(by="influence_score", ascending=False)
    
    with open(f"results/{cfg.experiment_name}/norm_reg_inverse.json", "w") as file:
        json.dump(norms, file, indent=4)

    if cfg.remove.enabled:
        print(f"\n>>> Neighrboring Dataset (Samples: {cfg.remove.num_samples_to_test})...")

        indices_to_remove = np.random.choice(len(train_set), cfg.remove.num_samples_to_test, replace=False)
        for i, idx in enumerate(indices_to_remove):
            print(f"[{i+1}/{cfg.remove.num_samples_to_test}] Removing Index {idx}...")
            del model
            torch.cuda.empty_cache()
            model = Net(input_dim, cfg.model.hidden_dim).to(device)

            model, remove_stats, hessian = train_and_measure(cfg, model, train_set, exclude_idx=int(idx))
            
            # matrix = hessian.detach().cpu().numpy()
            # np.savetxt(f'results/{cfg.experiment_name}/hessian_{idx}.txt', matrix, fmt='%.4f', delimiter='\t')
            np.savetxt(f'results/{cfg.experiment_name}/eigenvalue_{idx}.txt', remove_stats['eigenvalues'].detach().cpu().numpy(), fmt='%.4f', delimiter='\t')

            if cfg.analysis.use_exact_hessian:
                print("Plotting Eigenvalue Statistics:")
                plot_eigenvalues(remove_stats['eigenvalues'],
                                f'results/{cfg.experiment_name}/eigen_remove_idx_{idx}.png')

            # Compile Row
            row = {'sample_idx': idx}

            # Add all stats from the stats dict
            for key in ['max_eig', 'min_eig', 'trace', 'condition_number', 'mean_eig', 'median_eig', 'neg_eig_ratio']:
                if key in remove_stats:
                    row[key] = remove_stats[key]

                    # Calculate Deltas (Metric - Baseline)
                    # Note: Delta is meaningless if value is NaN
                    if key in base_stats and not np.isnan(remove_stats[key]):
                        row[f'delta_{key}'] = remove_stats[key] - base_stats[key]

            results.append(row)

            H_inv = torch.linalg.inv(hessian + cfg.analysis.reg * torch.eye(hessian.shape[0], device=device))
            influences, _ = compute_influence_vs_validation(
                model, if_loader, val_sample, nn.CrossEntropyLoss(), H_inv, device
            )

            # 6. Analysis & Visualization
            df = pd.DataFrame(influences)
            df.to_csv(f"results/{cfg.experiment_name}/if_remove_idx_{idx}.csv", index=False)

    
        df_eigen = pd.DataFrame(results)

        # Reorder columns for readability
        cols = ['sample_idx'] + [c for c in df_eigen.columns if c != 'sample_idx']
        df_eigen = df_eigen[cols]

        csv_path = f"results/{cfg.experiment_name}/remove_hessian_metrics.csv"
        df_eigen.to_csv(csv_path, index=False)
        print(f"\n>>> Experiment Complete.")
        print(f"Saved metrics to: {csv_path}")
        print(df_eigen.head())


if __name__ == "__main__":
    run_influence_analysis()