import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import os

def analyze_weights(file_paths, layers_of_interest, output_dir="analysis_plots"):
    """
    Loads weight distribution data from JSON files, calculates statistics,
    and generates histogram plots for specified layers.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Group files by trajectory (epoch vs. step)
    trajectories = {}
    for path in file_paths:
        filename = os.path.basename(path)
        prefix = filename.split('_')[1]
        if prefix not in trajectories:
            trajectories[prefix] = []
        trajectories[prefix].append(path)

    for prefix, files in trajectories.items():
        print(f"--- Analyzing Trajectory: '{prefix}' ---")
        # Sort files numerically by step/epoch number
        files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))) if ''.join(filter(str.isdigit, os.path.basename(x))) else float('inf'))

        for layer in layers_of_interest:
            plt.figure(figsize=(12, 7))
            plt.suptitle(f"Weight Distribution for Layer: {layer} (Trajectory: {prefix})", fontsize=16)
            
            print(f"\nLayer: {layer}")
            print("-" * 60)
            print(f"{'File':<32} | {'Mean':>10} | {'Std Dev':>10} | {'Skew':>8} | {'Kurtosis':>8}")
            print("-" * 60)

            for file_path in files:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                params = data[0]['parameters']
                if layer in params:
                    weights = np.array(params[layer]['values'])
                    
                    # Stats
                    mean_val = np.mean(weights)
                    std_val = np.std(weights)
                    skew_val = skew(weights)
                    kurt_val = kurtosis(weights)
                    
                    filename = os.path.basename(file_path)
                    print(f"{filename:<32} | {mean_val:>10.5f} | {std_val:>10.5f} | {skew_val:>8.4f} | {kurt_val:>8.4f}")

                    # Plotting
                    label = filename.replace('.json', '')
                    plt.hist(weights, bins=100, density=True, alpha=0.6, label=label)

            plt.title("Distribution Over Time")
            plt.xlabel("Weight Value")
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_filename = f"distribution_{prefix}_{layer}.png"
            plt.savefig(os.path.join(output_dir, plot_filename))
            plt.close()
            print(f"\nGenerated plot: {os.path.join(output_dir, plot_filename)}")
            print("-" * 60)


if __name__ == "__main__":
    # Find all relevant JSON files in the data directory
    weight_files = [
        "data/weights_epoch_0.json",
        "data/weights_epoch_1.json",
        "data/weights_epoch_final.json",
        "data/weights_step_0.json",
        "data/weights_step_11.json"
    ]
    
    # Specify the layers you want to analyze
    layers = [
        "conv1.weight",
        "conv2.weight",
        "fc1.weight",
        "fc2.weight"
    ]
    
    analyze_weights(weight_files, layers)
