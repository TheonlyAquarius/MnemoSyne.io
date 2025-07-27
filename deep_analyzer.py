import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, wasserstein_distance
from sklearn.decomposition import PCA
import os
import re
import json
from collections import defaultdict
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Create output directories
os.makedirs('deep_analysis', exist_ok=True)
os.makedirs('deep_analysis/plots', exist_ok=True)
os.makedirs('deep_analysis/reports', exist_ok=True)

class ComprehensiveWeightAnalyzer:
    def __init__(self):
        self.weight_data = {}
        self.trajectory_data = {}
        self.layers = ["conv1.weight", "conv2.weight", "fc1.weight", "fc2.weight"]
        self.report = {
            "summary": {},
            "anomalies": [],
            "patterns": [],
            "critical_transitions": []
        }
        
    def load_all_weights(self):
        """Load all .pth weight files from both checkpoint directories"""
        checkpoint_dirs = [
            "checkpoints_weights_cnn",
            "generalized_trajectory_weights"
        ]
        
        for directory in checkpoint_dirs:
            if not os.path.exists(directory):
                print(f"Warning: Directory {directory} does not exist")
                continue
                
            for file in os.listdir(directory):
                if file.endswith('.pth'):
                    file_path = os.path.join(directory, file)
                    self._load_single_checkpoint(file_path, directory)
    
    def _load_single_checkpoint(self, file_path, source_dir):
        """Load a single checkpoint file and extract weight data"""
        try:
            # Determine trajectory type
            if 'checkpoints_weights_cnn' in source_dir:
                trajectory_type = 'epoch'
                # Extract epoch number
                match = re.search(r'epoch_(\d+|final)', os.path.basename(file_path))
                if match:
                    step_id = match.group(1)
                    if step_id == 'final':
                        step_id = 999  # Special value for final
                    else:
                        step_id = int(step_id)
                else:
                    step_id = 0
            else:  # generalized_trajectory_weights
                trajectory_type = 'step'
                # Extract step number
                match = re.search(r'step_(\d+)', os.path.basename(file_path))
                step_id = int(match.group(1)) if match else 0
            
            # Load checkpoint
            checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint.state_dict()
            
            # Store weight data
            key = f"{trajectory_type}_{step_id}"
            self.weight_data[key] = {
                'trajectory': trajectory_type,
                'step': step_id,
                'weights': {}
            }
            
            for layer in self.layers:
                if layer in state_dict:
                    self.weight_data[key]['weights'][layer] = state_dict[layer].cpu().numpy()
            
            print(f"Loaded {key}: {len(self.weight_data[key]['weights'])} layers")
            
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    def organize_trajectories(self):
        """Organize weight data by trajectory type"""
        self.trajectory_data = {
            'epoch': {},
            'step': {}
        }
        
        for key, data in self.weight_data.items():
            traj_type = data['trajectory']
            step = data['step']
            self.trajectory_data[traj_type][step] = data['weights']
    
    def calculate_basic_statistics(self):
        """Calculate basic statistical measures for all weights"""
        stats = {}
        
        for key, data in self.weight_data.items():
            stats[key] = {}
            for layer, weights in data['weights'].items():
                flat_weights = weights.flatten()
                stats[key][layer] = {
                    'mean': float(np.mean(flat_weights)),
                    'std': float(np.std(flat_weights)),
                    'skew': float(skew(flat_weights)),
                    'kurtosis': float(kurtosis(flat_weights)),
                    'min': float(np.min(flat_weights)),
                    'max': float(np.max(flat_weights)),
                    'median': float(np.median(flat_weights))
                }
        
        return stats
    
    def detect_anomalies(self):
        """Detect statistical anomalies in weight evolution"""
        anomalies = []
        
        for layer in self.layers:
            # Collect statistics over time for each layer
            layer_stats = []
            time_points = []
            
            for key, data in sorted(self.weight_data.items()):
                if layer in data['weights']:
                    weights = data['weights'][layer].flatten()
                    layer_stats.append({
                        'mean': np.mean(weights),
                        'std': np.std(weights),
                        'skew': skew(weights),
                        'kurtosis': kurtosis(weights)
                    })
                    time_points.append(key)
            
            if len(layer_stats) < 3:
                continue
                
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(layer_stats)
            time_points = np.array(time_points)
            
            # Detect outliers using IQR method for each statistic
            for stat in ['mean', 'std', 'skew', 'kurtosis']:
                Q1 = df[stat].quantile(0.25)
                Q3 = df[stat].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[stat] < lower_bound) | (df[stat] > upper_bound)]
                for idx in outliers.index:
                    anomalies.append({
                        'layer': layer,
                        'statistic': stat,
                        'value': float(outliers.iloc[idx][stat]),
                        'time_point': time_points[idx],
                        'type': 'statistical_outlier'
                    })
        
        self.report['anomalies'].extend(anomalies)
        return anomalies
    
    def calculate_wasserstein_distances(self):
        """Calculate Wasserstein distances between consecutive weight distributions"""
        w_distances = {}
        
        for layer in self.layers:
            w_distances[layer] = {'epoch': [], 'step': []}
            
            for traj_type in ['epoch', 'step']:
                steps = sorted([k for k, v in self.trajectory_data[traj_type].items() if layer in v])
                
                for i in range(len(steps) - 1):
                    step1, step2 = steps[i], steps[i+1]
                    weights1 = self.trajectory_data[traj_type][step1][layer].flatten()
                    weights2 = self.trajectory_data[traj_type][step2][layer].flatten()
                    
                    # Subsample if arrays are too large
                    if len(weights1) > 10000:
                        indices = np.random.choice(len(weights1), 10000, replace=False)
                        weights1 = weights1[indices]
                        weights2 = weights2[indices]
                    
                    try:
                        distance = wasserstein_distance(weights1, weights2)
                        w_distances[layer][traj_type].append({
                            'step1': step1,
                            'step2': step2,
                            'distance': distance
                        })
                    except Exception as e:
                        print(f"Error calculating Wasserstein distance for {layer} {step1}->{step2}: {e}")
        
        return w_distances
    
    def analyze_spectral_properties(self):
        """Analyze spectral properties of weight matrices"""
        spectral_data = {}
        
        for key, data in self.weight_data.items():
            spectral_data[key] = {}
            for layer, weights in data['weights'].items():
                try:
                    # Handle different weight shapes
                    if len(weights.shape) > 2:  # Convolutional layers
                        c_out, c_in, k_h, k_w = weights.shape
                        weights_matrix = weights.reshape(c_out, -1)
                    else:  # Fully connected layers
                        weights_matrix = weights
                    
                    # Compute eigenvalues
                    if weights_matrix.shape[0] <= weights_matrix.shape[1]:
                        cov_matrix = weights_matrix @ weights_matrix.T
                    else:
                        cov_matrix = weights_matrix.T @ weights_matrix
                    
                    eigenvalues = np.linalg.eigvalsh(cov_matrix)
                    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
                    
                    # Calculate spectral statistics
                    spectral_data[key][layer] = {
                        'eigenvalues': eigenvalues[:min(100, len(eigenvalues))].tolist(),  # Top 100
                        'spectral_radius': float(np.max(eigenvalues)),
                        'spectral_gap': float(eigenvalues[0] - eigenvalues[1]) if len(eigenvalues) > 1 else 0,
                        'effective_rank': float(np.sum(eigenvalues > (np.max(eigenvalues) * 0.01)))
                    }
                except Exception as e:
                    print(f"Error in spectral analysis for {key} {layer}: {e}")
                    spectral_data[key][layer] = {
                        'eigenvalues': [],
                        'spectral_radius': 0,
                        'spectral_gap': 0,
                        'effective_rank': 0
                    }
        
        return spectral_data
    
    def detect_critical_transitions(self):
        """Detect critical transitions in training using variance-based methods"""
        transitions = []
        
        for layer in self.layers:
            for traj_type in ['epoch', 'step']:
                steps = sorted([k for k, v in self.trajectory_data[traj_type].items() if layer in v])
                if len(steps) < 5:  # Need enough points for meaningful analysis
                    continue
                    
                # Calculate sliding window variance
                window_size = max(3, len(steps) // 4)
                variances = []
                time_points = []
                
                for i in range(len(steps) - window_size + 1):
                    window_steps = steps[i:i+window_size]
                    window_weights = []
                    
                    for step in window_steps:
                        weights = self.trajectory_data[traj_type][step][layer].flatten()
                        # Subsample for efficiency
                        if len(weights) > 5000:
                            indices = np.random.choice(len(weights), 5000, replace=False)
                            weights = weights[indices]
                        window_weights.extend(weights)
                    
                    variances.append(np.var(window_weights))
                    # Use middle point of window as time reference
                    mid_idx = i + window_size // 2
                    time_points.append(steps[mid_idx])
                
                if len(variances) < 3:
                    continue
                
                # Detect variance peaks (potential critical transitions)
                var_array = np.array(variances)
                mean_var = np.mean(var_array)
                std_var = np.std(var_array)
                
                # Find points with variance > mean + 2*std (outliers)
                threshold = mean_var + 2 * std_var
                peak_indices = np.where(var_array > threshold)[0]
                
                for idx in peak_indices:
                    transitions.append({
                        'layer': layer,
                        'trajectory': traj_type,
                        'time_point': time_points[idx],
                        'variance': float(var_array[idx]),
                        'type': 'variance_peak'
                    })
        
        self.report['critical_transitions'].extend(transitions)
        return transitions
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        # 1. Evolution of statistical measures
        self._plot_statistical_evolution()
        
        # 2. Wasserstein distance evolution
        self._plot_wasserstein_distances()
        
        # 3. Spectral properties evolution
        self._plot_spectral_evolution()
        
        # 4. Distribution comparison heatmaps
        self._plot_distribution_heatmaps()
        
        # 5. Anomaly visualization
        self._plot_anomalies()
    
    def _plot_statistical_evolution(self):
        """Plot evolution of basic statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        stats = ['mean', 'std', 'skew', 'kurtosis']
        colors = {'epoch': 'blue', 'step': 'red'}
        
        for i, stat in enumerate(stats):
            for traj_type in ['epoch', 'step']:
                x_vals, y_vals = [], []
                for key, data in sorted(self.weight_data.items()):
                    if data['trajectory'] == traj_type and 'conv1.weight' in data['weights']:
                        x_vals.append(data['step'])
                        flat_weights = data['weights']['conv1.weight'].flatten()
                        if stat == 'mean':
                            y_vals.append(np.mean(flat_weights))
                        elif stat == 'std':
                            y_vals.append(np.std(flat_weights))
                        elif stat == 'skew':
                            y_vals.append(skew(flat_weights))
                        elif stat == 'kurtosis':
                            y_vals.append(kurtosis(flat_weights))
                
                if x_vals:
                    axes[i].plot(x_vals, y_vals, 'o-', color=colors[traj_type], 
                                label=traj_type.capitalize(), alpha=0.7)
                    axes[i].set_title(f'{stat.capitalize()} Evolution')
                    axes[i].set_xlabel('Training Step')
                    axes[i].set_ylabel(stat.capitalize())
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('deep_analysis/plots/statistical_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_wasserstein_distances(self):
        """Plot Wasserstein distances between consecutive steps"""
        w_distances = self.calculate_wasserstein_distances()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        colors = {'epoch': 'blue', 'step': 'red'}
        
        for i, layer in enumerate(self.layers):
            for traj_type in ['epoch', 'step']:
                if w_distances[layer][traj_type]:
                    # Use step2 (the destination step) for x-axis
                    steps = [d['step2'] for d in w_distances[layer][traj_type]]
                    distances = [d['distance'] for d in w_distances[layer][traj_type]]
                    axes[i].plot(steps, distances, 'o-', color=colors[traj_type],
                                label=f'{traj_type.capitalize()}', alpha=0.7)
            
            axes[i].set_title(f'{layer} - Wasserstein Distance')
            axes[i].set_xlabel('Training Step')
            axes[i].set_ylabel('Wasserstein Distance')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('deep_analysis/plots/wasserstein_distances.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_spectral_evolution(self):
        """Plot evolution of spectral properties"""
        spectral_data = self.analyze_spectral_properties()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        metrics = ['spectral_radius', 'spectral_gap', 'effective_rank']
        colors = {'epoch': 'blue', 'step': 'red'}
        
        for i, metric in enumerate(metrics):
            for traj_type in ['epoch', 'step']:
                x_vals, y_vals = [], []
                for key, data in sorted(spectral_data.items()):
                    trajectory = self.weight_data[key]['trajectory']
                    step = self.weight_data[key]['step']
                    
                    if trajectory == traj_type and 'conv1.weight' in data and data['conv1.weight'][metric] is not None:
                        x_vals.append(step)
                        y_vals.append(data['conv1.weight'][metric])
                
                if x_vals:
                    axes[i].plot(x_vals, y_vals, 'o-', color=colors[traj_type],
                                label=traj_type.capitalize(), alpha=0.7)
                    axes[i].set_title(f'{metric.replace("_", " ").title()}')
                    axes[i].set_xlabel('Training Step')
                    axes[i].set_ylabel(metric.replace("_", " ").title())
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
        
        # Plot eigenvalue distribution for final step
        final_key = max([k for k, v in spectral_data.items() if 'conv1.weight' in v and v['conv1.weight']['eigenvalues']],
                       key=lambda x: self.weight_data[x]['step'])
        if final_key in spectral_data and 'conv1.weight' in spectral_data[final_key]:
            eigenvals = spectral_data[final_key]['conv1.weight']['eigenvalues']
            if len(eigenvals) > 0:
                axes[3].semilogy(eigenvals, 'b-')
                axes[3].set_title('Eigenvalue Spectrum (Final Step)')
                axes[3].set_xlabel('Index')
                axes[3].set_ylabel('Eigenvalue (log scale)')
                axes[3].grid(True, alpha=0.3)
            else:
                axes[3].text(0.5, 0.5, 'No eigenvalues available', ha='center', va='center', transform=axes[3].transAxes)
                axes[3].set_title('Eigenvalue Spectrum (Final Step)')
        else:
            axes[3].text(0.5, 0.5, 'No spectral data available', ha='center', va='center', transform=axes[3].transAxes)
            axes[3].set_title('Eigenvalue Spectrum (Final Step)')
        
        plt.tight_layout()
        plt.savefig('deep_analysis/plots/spectral_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_distribution_heatmaps(self):
        """Create heatmaps of weight distributions"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Get weight distributions for first and last steps
        steps = sorted([k for k in self.weight_data.keys()])
        if len(steps) >= 2:
            for i, step_key in enumerate([steps[0], steps[-1]]):
                data = self.weight_data[step_key]
                all_weights = []
                layer_names = []
                
                for layer in self.layers:
                    if layer in data['weights']:
                        weights = data['weights'][layer].flatten()
                        # Subsample if too large
                        if len(weights) > 10000:
                            indices = np.random.choice(len(weights), 10000, replace=False)
                            weights = weights[indices]
                        all_weights.append(weights)
                        layer_names.append(layer)
                
                if all_weights:
                    # Create 2D histogram for each layer
                    max_len = max(len(w) for w in all_weights)
                    padded_weights = []
                    for weights in all_weights:
                        if len(weights) < max_len:
                            # Pad with NaN
                            padded = np.full(max_len, np.nan)
                            padded[:len(weights)] = weights
                            padded_weights.append(padded)
                        else:
                            padded_weights.append(weights)
                    
                    weights_matrix = np.array(padded_weights)
                    im = axes[i].imshow(weights_matrix, aspect='auto', cmap='viridis')
                    axes[i].set_title(f'Weight Distributions - {step_key}')
                    axes[i].set_xlabel('Weight Index (subsampled)')
                    axes[i].set_ylabel('Layer')
                    axes[i].set_yticks(range(len(layer_names)))
                    axes[i].set_yticklabels(layer_names)
                    plt.colorbar(im, ax=axes[i])
        else:
            for i in range(2):
                axes[i].text(0.5, 0.5, 'Insufficient data\nfor heatmap', ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title('Weight Distributions')
        
        plt.tight_layout()
        plt.savefig('deep_analysis/plots/distribution_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_anomalies(self):
        """Visualize detected anomalies"""
        anomalies = self.detect_anomalies()
        if not anomalies:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, 'No anomalies detected', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Anomalies Visualization')
            plt.savefig('deep_analysis/plots/anomalies.png', dpi=300, bbox_inches='tight')
            plt.close()
            return
            
        # Convert to DataFrame for easier handling
        df_anomalies = pd.DataFrame(anomalies)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        stats = ['mean', 'std', 'skew', 'kurtosis']
        colors = {'epoch': 'blue', 'step': 'red'}
        
        for i, stat in enumerate(stats):
            stat_anomalies = df_anomalies[df_anomalies['statistic'] == stat]
            if stat_anomalies.empty:
                axes[i].text(0.5, 0.5, f'No {stat} anomalies', ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{stat.capitalize()} Anomalies')
                continue
                
            # Plot normal evolution for context
            for traj_type in ['epoch', 'step']:
                x_vals, y_vals = [], []
                for key, data in sorted(self.weight_data.items()):
                    if data['trajectory'] == traj_type and 'conv1.weight' in data['weights']:
                        x_vals.append(data['step'])
                        flat_weights = data['weights']['conv1.weight'].flatten()
                        if stat == 'mean':
                            y_vals.append(np.mean(flat_weights))
                        elif stat == 'std':
                            y_vals.append(np.std(flat_weights))
                        elif stat == 'skew':
                            y_vals.append(skew(flat_weights))
                        elif stat == 'kurtosis':
                            y_vals.append(kurtosis(flat_weights))
                
                if x_vals:
                    axes[i].plot(x_vals, y_vals, '-', color=colors[traj_type], 
                                label=f'{traj_type.capitalize()} (normal)', alpha=0.5)
            
            # Highlight anomalies
            for _, anomaly in stat_anomalies.iterrows():
                if anomaly['layer'] == 'conv1.weight':
                    # Extract step from time_point
                    time_parts = anomaly['time_point'].split('_')
                    step = int(time_parts[1]) if time_parts[1].isdigit() else 0
                    axes[i].plot(step, anomaly['value'], 'ro', markersize=10, 
                                label='Anomaly' if 'Anomaly' not in [t.get_label() for t in axes[i].get_legend().get_texts()] else "")
            
            axes[i].set_title(f'{stat.capitalize()} Anomalies')
            axes[i].set_xlabel('Training Step')
            axes[i].set_ylabel(stat.capitalize())
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('deep_analysis/plots/anomalies.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        # Basic statistics
        stats = self.calculate_basic_statistics()
        
        # Anomalies
        anomalies = self.detect_anomalies()
        
        # Critical transitions
        transitions = self.detect_critical_transitions()
        
        # Create summary
        total_checkpoints = len(self.weight_data)
        layers_analyzed = len(self.layers)
        
        self.report['summary'] = {
            'total_checkpoints_analyzed': total_checkpoints,
            'layers_analyzed': layers_analyzed,
            'anomalies_detected': len(anomalies),
            'critical_transitions_detected': len(transitions),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Save report to JSON
        with open('deep_analysis/reports/comprehensive_analysis.json', 'w') as f:
            json.dump(self.report, f, indent=2)
        
        # Generate text report
        report_text = f"""
# Comprehensive Neural Network Weight Analysis Report

## Summary
- Total checkpoints analyzed: {total_checkpoints}
- Layers analyzed: {layers_analyzed}
- Anomalies detected: {len(anomalies)}
- Critical transitions detected: {len(transitions)}
- Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Key Findings

### Statistical Patterns
1. Weight distribution characteristics across training
2. Evolution of higher-order moments (skewness, kurtosis)
3. Comparison between epoch-based and step-based trajectories

### Anomalies Detected
{len(anomalies)} statistical outliers were identified:
"""
        for anomaly in anomalies[:10]:  # Show first 10
            report_text += f"- {anomaly['layer']} {anomaly['statistic']}: {anomaly['value']:.4f} at {anomaly['time_point']}\n"
        if len(anomalies) > 10:
            report_text += f"... and {len(anomalies) - 10} more anomalies\n"

        report_text += f"""

### Critical Transitions
{len(transitions)} potential critical transitions were identified:
"""
        for transition in transitions[:10]:  # Show first 10
            report_text += f"- {transition['layer']} {transition['trajectory']}: variance {transition['variance']:.4f} at step {transition['time_point']}\n"
        if len(transitions) > 10:
            report_text += f"... and {len(transitions) - 10} more transitions\n"

        report_text += """

## Visualizations Generated
1. Statistical evolution plots (mean, std, skew, kurtosis)
2. Wasserstein distance evolution
3. Spectral properties evolution
4. Weight distribution heatmaps
5. Anomaly visualization

## Methodology
This analysis employed multiple advanced techniques:
- Statistical moment analysis (mean, variance, skewness, kurtosis)
- Wasserstein distance for distribution comparison
- Spectral analysis of weight matrices
- Variance-based critical transition detection
- Outlier detection using IQR method

All visualizations and detailed data are available in the 'deep_analysis' directory.
"""

        with open('deep_analysis/reports/comprehensive_analysis.txt', 'w') as f:
            f.write(report_text)
        
        print("Comprehensive analysis report generated:")
        print("- deep_analysis/reports/comprehensive_analysis.json")
        print("- deep_analysis/reports/comprehensive_analysis.txt")
        print("- deep_analysis/plots/ (various visualization files)")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting comprehensive weight analysis...")
        
        # Load all weights
        print("1. Loading weight checkpoints...")
        self.load_all_weights()
        self.organize_trajectories()
        
        if not self.weight_data:
            print("No weight data found. Exiting.")
            return
        
        print(f"   Loaded {len(self.weight_data)} checkpoints")
        
        # Generate visualizations
        print("2. Generating visualizations...")
        self.generate_visualizations()
        
        # Detect anomalies
        print("3. Detecting anomalies...")
        anomalies = self.detect_anomalies()
        print(f"   Found {len(anomalies)} anomalies")
        
        # Detect critical transitions
        print("4. Detecting critical transitions...")
        transitions = self.detect_critical_transitions()
        print(f"   Found {len(transitions)} critical transitions")
        
        # Generate report
        print("5. Generating comprehensive report...")
        self.generate_report()
        
        print("\nAnalysis complete! Check the 'deep_analysis' directory for results.")

# Run the analysis
if __name__ == "__main__":
    analyzer = ComprehensiveWeightAnalyzer()
    analyzer.run_complete_analysis()
