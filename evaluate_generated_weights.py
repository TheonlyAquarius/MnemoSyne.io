import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import glob

# --- Model & Utility Imports ---
from target_cnn import TargetCNN
from train_vae_and_capture_checkpoints import IntelligentVAE, truly_analyze_dataset
from diffusion_model import WholeVectorPerceiver, flatten_state_dict, unflatten_to_state_dict, get_target_model_flat_dim

# --- Evaluation Functions ---

def evaluate_cnn_performance(model, test_loader, device, criterion):
    model.eval()
    test_loss, correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item() * data.size(0)
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)
    return 100. * correct / total_samples, test_loss / total_samples

def evaluate_vae_performance(model, test_loader, device, criterion):
    model.eval()
    test_loss, total_samples = 0, 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon_batch, _, _ = model(data)
            loss = criterion(recon_batch, data)
            test_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
    return 0.0, test_loss / total_samples # Return 0 for accuracy as it's not applicable

# --- Core Logic ---

def generate_checkpoints(diffusion_model, initial_weights_flat, num_steps, ref_state_dict, device):
    print(f"Generating checkpoints for {num_steps} steps...")
    generated_weights = []
    current_weights = initial_weights_flat.to(device)
    diffusion_model.to(device).eval()
    for t_idx in range(num_steps):
        timestep = torch.tensor([[float(t_idx)]], device=device)
        with torch.no_grad():
            next_weights = diffusion_model(current_weights.unsqueeze(0), timestep).squeeze(0)
        generated_weights.append(next_weights.cpu())
        current_weights = next_weights
        if (t_idx + 1) % 5 == 0 or (t_idx + 1) == num_steps:
            print(f"  Generated step {t_idx+1}/{num_steps}")
    return generated_weights

def get_user_choice(options, prompt):
    print(prompt)
    for i, (key, val) in enumerate(options.items()):
        print(f"  {i+1}) {val['description']}")
    while True:
        try:
            choice = int(input(f"Enter your choice (1-{len(options)}): "))
            if 1 <= choice <= len(options):
                return list(options.keys())[choice - 1]
        except ValueError:
            pass
        print("Invalid input, please try again.")

def discover_setups():
    setups = {}
    diffusion_files = glob.glob("diffusion_optimizer*.pth")
    for df in diffusion_files:
        suffix = df.replace("diffusion_optimizer", "").replace(".pth", "")
        traj_dir = f"checkpoints_weights{suffix}"
        if os.path.isdir(traj_dir):
            model_type = 'VAE' if 'vae' in suffix else 'CNN'
            setups[suffix[1:] if suffix.startswith('_') else suffix] = {
                'description': f"Target: {model_type}, Diffusion Model: {df}, checkpoints: {traj_dir}",
                'model_type': model_type,
                'diffusion_path': df,
                'checkpoints_dir': traj_dir
            }
    return setups

# --- Main Execution ---

def main():
    setups = discover_setups()
    if not setups:
        print("Error: No valid evaluation setups found.")
        print("Please ensure you have a trained diffusion model (e.g., 'diffusion_optimizer_cnn.pth')")
        print("and its corresponding checkpoints directory (e.g., 'checkpoints_weights_cnn/').")
        return

    if len(setups) == 1:
        system_key = list(setups.keys())[0]
        print(f"Found one setup: {setups[system_key]['description']}")
    else:
        system_key = get_user_choice(setups, "Please choose the target system to evaluate:")
    
    chosen_setup = setups[system_key]
    model_type = chosen_setup['model_type']

    experiments = {
        'replicate': {'description': "Replicate Original checkpoints (denoise known initial weights)"},
        'generalize': {'description': "Test Generalization (denoise NEW random initial weights)"}
    }
    experiment_type = get_user_choice(experiments, "\nPlease choose the experiment to run:")

    print(f"\n----- Starting Evaluation ----- ")
    print(f"  Target Model: {model_type}")
    print(f"  Experiment: {experiments[experiment_type]['description']}")
    print("-----------------------------\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Select and instantiate the target model
    if model_type == 'CNN':
        target_model = TargetCNN()
        eval_function = evaluate_cnn_performance
        criterion = nn.CrossEntropyLoss(reduction='sum')
        perf_metric, loss_metric = "Accuracy (%)", "Avg Loss"
    else: # VAE
        analysis = truly_analyze_dataset()
        target_model = IntelligentVAE(analysis)
        eval_function = evaluate_vae_performance
        criterion = nn.MSELoss(reduction='sum')
        perf_metric, loss_metric = "N/A", "Reconstruction Loss"

    ref_state_dict = target_model.state_dict()
    flat_dim = get_target_model_flat_dim(ref_state_dict)

    # 2. Load the corresponding diffusion model
    diffusion_model = WholeVectorPerceiver(
        flat_dim=flat_dim,
        latent_dim=512,
        num_latents=64,
        depth=6,
    )
    diffusion_model.load_state_dict(torch.load(chosen_setup['diffusion_path'], map_location=device))

    # 3. Define initial weights based on experiment type
    if experiment_type == 'replicate':
        initial_weights_path = os.path.join(chosen_setup['checkpoints_dir'], "weights_epoch_0.pth")
        initial_state_dict = torch.load(initial_weights_path, map_location='cpu')
        print(f"Using initial weights from: {initial_weights_path}")
    else: # generalize
        print("Generating new, random initial weights for generalization test.")
        new_random_model = target_model.__class__() if model_type == 'CNN' else IntelligentVAE(truly_analyze_dataset())
        initial_state_dict = new_random_model.state_dict()

    initial_weights_flat = flatten_state_dict(initial_state_dict)

    # 4. Generate the checkpoints
    num_steps = len(glob.glob(os.path.join(chosen_setup['checkpoints_dir'], "weights_epoch_*.pth"))) - 1
    generated_weights = generate_checkpoints(diffusion_model, initial_weights_flat, num_steps, ref_state_dict, device)

    # 5. Evaluate the generated checkpoints
    print("\nEvaluating performance along the generated checkpoints...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_loader = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=transform), batch_size=256)
    
    eval_model = target_model.__class__().to(device) if model_type == 'CNN' else IntelligentVAE(truly_analyze_dataset()).to(device)

    results = []
    # Evaluate initial state
    eval_model.load_state_dict(initial_state_dict)
    perf, loss = eval_function(eval_model, test_loader, device, criterion)
    results.append((perf, loss))
    print(f"Step 0 (Initial Weights): {perf_metric.split(' ')[0]} = {perf:.2f}, {loss_metric} = {loss:.4f}")

    # Evaluate generated states
    for i, weights in enumerate(generated_weights):
        state_dict = unflatten_to_state_dict(weights, ref_state_dict)
        eval_model.load_state_dict(state_dict)
        perf, loss = eval_function(eval_model, test_loader, device, criterion)
        results.append((perf, loss))
        print(f"Generated Step {i+1}/{num_steps}: {perf_metric.split(' ')[0]} = {perf:.2f}, {loss_metric} = {loss:.4f}")

    # 6. Plot the results
    print("\nPlotting results...")
    performances, losses = zip(*results)
    plt.figure(figsize=(12, 6))
    plt.plot(losses, label=f"Generated {loss_metric}", marker='o', color='tab:red')
    plt.xlabel("Generative Step")
    plt.ylabel(loss_metric, color='tab:red')
    plt.tick_params(axis='y', labelcolor='tab:red')
    plt.title(f"{model_type} Performance from {experiment_type.capitalize()} Weights")
    plt.grid(True)

    if model_type == 'CNN':
        ax2 = plt.gca().twinx()
        ax2.plot(performances, label=f"Generated {perf_metric}", marker='x', color='tab:blue')
        ax2.set_ylabel(perf_metric, color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.tight_layout()
    plot_path = f"{model_type.lower()}_{experiment_type}_evaluation.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == '__main__':
    main()