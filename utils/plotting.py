import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_eigenvalues(eigenvalues, save_path="eigenvalue_spectrum.png"):
    """
    Plots the sorted eigenvalues and their density histogram.
    """
    # Convert to numpy
    eigs = eigenvalues.detach().cpu().numpy()
    
    # Sort descending
    eigs = np.sort(eigs)[::-1]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: The Spectrum (Magnitude vs Index)
    axes[0].plot(eigs)
    axes[0].set_title("Eigenvalue Spectrum (Sorted)")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Eigenvalue")
    # axes[0].set_xscale('log')
    axes[0].set_yscale('symlog', linthresh=0.01)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Density (Histogram)
    axes[1].hist(eigs, bins=len(eigs)//10, color='orange', alpha=0.7)
    axes[1].set_title("Eigenvalue Density (Log Scale)")
    axes[1].set_xlabel("Eigenvalue Magnitude")
    axes[1].set_ylabel("Frequency")
    axes[1].set_yscale('log')
    axes[1].set_xscale('symlog', linthresh=0.01)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()