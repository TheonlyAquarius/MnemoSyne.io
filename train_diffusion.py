import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import glob
import numpy as np

from diffusion_model import SimpleWeightSpaceDiffusion, flatten_state_dict, get_target_model_flat_dim
from target_cnn import TargetCNN # To get a reference state_dict for unflattening

class WeightcheckpointsDataset(Dataset):
    def __init__(self, checkpoints_dir, reference_state_dict, max_checkpoints_len=None):
        """
        Dataset for loading pairs of (current_weights, next_weights) from a saved checkpoints.
        The "forward" process of the diffusion model learns to reverse the optimization.
        So, if optimization goes W_t -> W_{t+1} (less loss),
        the diffusion's "noising" learns W_{t+1} -> W_t.
        The diffusion model itself will be trained to predict W_{t+1} given W_t and t.

        Args:
            checkpoints_dir (str): Directory containing saved .pth weight files.
            reference_state_dict (dict): A sample state_dict from the target model,
                                         used to know the structure for unflattening if needed,
                                         and to determine flat_dim.
            max_checkpoints_len (int, optional): If set, limits the number of checkpoints steps used.
        """
        self.checkpoints_dir = checkpoints_dir
        self.reference_state_dict = reference_state_dict
        self.flat_dim = get_target_model_flat_dim(reference_state_dict)

        weight_files = sorted(
            glob.glob(os.path.join(checkpoints_dir, "weights_epoch_*.pth")),
            key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else -1
        )
        # Ensure 'weights_epoch_00.pth' is first if it exists, then numeric epochs.
        # 'weights_epoch_final.pth' is usually the same as the last numeric epoch, handle as needed.
        # For this setup, we assume numeric epoch files are the primary sequence.

        if not weight_files:
            raise FileNotFoundError(f"No weight files found in {checkpoints_dir}. Run train_target_model.py first.")

        print(f"Found {len(weight_files)} weight files.")

        self.weight_files = weight_files
        if max_checkpoints_len:
            self.weight_files = self.weight_files[:max_checkpoints_len + 1] # Need one more file than transitions

        self.num_total_steps = len(self.weight_files)
        print(f"Found {self.num_total_steps} weight files for the checkpoints.")


    def __len__(self):
        # We have one less pair than the number of weight files
        return self.num_total_steps - 1

    def __getitem__(self, idx):
        # idx corresponds to the transition from state idx to state idx+1
        # Load the required weights on-the-fly
        w_path_current = self.weight_files[idx]
        w_path_next = self.weight_files[idx+1]

        try:
            state_dict_current = torch.load(w_path_current, map_location='cuda')
            current_w = flatten_state_dict(state_dict_current)

            state_dict_next = torch.load(w_path_next, map_location='cuda')
            target_next_w = flatten_state_dict(state_dict_next)

            if current_w.shape[0] != self.flat_dim or target_next_w.shape[0] != self.flat_dim:
                raise ValueError(f"Dimension mismatch in loaded weights for index {idx}.")

            # The timestep 't' is the index of the current state in the checkpoints
            t = torch.tensor([float(idx)])

            return current_w, target_next_w, t
        except Exception as e:
            print(f"Error loading weights for index {idx}: {e}")
            # Return dummy data or raise an error, depending on desired robustness
            # For simplicity, we'll raise it further.
            raise e


def train_diffusion_model(
    checkpoints_dir,
    target_model_ref_for_dims, # An instance of TargetCNN or its state_dict
    epochs=100,
    lr=0.001,
    batch_size=128,
    time_emb_dim=32,
    hidden_dim_diff_model=762,
    save_path="trained_diffusion_model.pth",
    max_traj_len_for_training=None
):
    print("Starting Diffusion Model training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if isinstance(target_model_ref_for_dims, nn.Module):
        reference_state_dict = target_model_ref_for_dims.state_dict()
    else: # assuming it's already a state_dict
        reference_state_dict = target_model_ref_for_dims

    target_flat_dim = get_target_model_flat_dim(reference_state_dict)
    print(f"Target model flattened dimension: {target_flat_dim}")

    # Initialize Diffusion Model
    diffusion_model = SimpleWeightSpaceDiffusion(
        target_model_flat_dim=target_flat_dim,
        time_emb_dim=time_emb_dim,
        hidden_dim=hidden_dim_diff_model
    ).to(device)
    print("Diffusion model initialized.")

    # Prepare Dataset and DataLoader
    print("Loading weight checkpoints dataset...")
    dataset = WeightcheckpointsDataset(checkpoints_dir, reference_state_dict, max_checkpoints_len=max_traj_len_for_training)
    if len(dataset) == 0:
        print("Dataset is empty. Cannot train. Ensure 'checkpoints_weights_cnn' has valid data.")
        return
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset loaded with {len(dataset)} pairs.")

    # Loss and Optimizer
    criterion = nn.MSELoss() # Predicts the next state directly
    optimizer = optim.Adam(diffusion_model.parameters(), lr=lr)
    print("Criterion and Optimizer initialized.")

    # Training Loop
    print("Starting training loop...")
    for epoch in range(1, epochs + 1):
        diffusion_model.train()
        total_loss = 0
        for current_weights_flat, target_next_weights_flat, timesteps_t in dataloader:
            current_weights_flat = current_weights_flat.to(device)
            target_next_weights_flat = target_next_weights_flat.to(device)
            timesteps_t = timesteps_t.to(device) # Shape [batch_size, 1]

            optimizer.zero_grad()

            # The diffusion model predicts W_{t+1} given W_t and t
            predicted_next_weights_flat = diffusion_model(current_weights_flat, timesteps_t)

            loss = criterion(predicted_next_weights_flat, target_next_weights_flat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch}/{epochs}], Average Loss: {avg_loss:.6f}")

    # Save the trained model
    torch.save(diffusion_model.state_dict(), save_path)
    print(f"Trained diffusion model saved to {save_path}")


if __name__ == '__main__':
    # --- Configuration ---
    # Directory where train_target_model.py saved the CNN's weight checkpoints
    CNN_checkpoints_DIR = "checkpoints_weights_cnn"

    # Number of epochs to train the diffusion model
    DIFFUSION_EPOCHS = 60 # Adjust as needed, more epochs generally better
    LEARNING_RATE_DIFFUSION = 0.0009
    BATCH_SIZE_DIFFUSION = 64 # Depends on memory and checkpoints length

    # Diffusion model architecture parameters
    TIME_EMB_DIM_DIFF = 64
    HIDDEN_DIM_DIFF = 762 # MLP hidden layer size

    # Path to save the trained diffusion model
    SAVED_DIFFUSION_MODEL_PATH = "diffusion_optimizer.pth"

    # Optional: Limit the number of checkpoints steps used for training (e.g., for faster iteration)
    # Set to None to use the full checkpoints.
    MAX_checkpoints_LENGTH_FOR_TRAINING = 49
    # Example: 10 would mean using only the first 10 transitions (W0->W1, W1->W2, ..., W9->W10)

    # --- End Configuration ---

    # We need a reference model or state_dict to know the dimensions and structure
    # Initialize a dummy TargetCNN for this purpose.
    # Ensure this matches the architecture used in train_target_model.py
    print("Initializing reference TargetCNN for dimensions...")
    reference_cnn = TargetCNN()
    # If checkpoints_weights_cnn was generated with a different model, this must match.

    # Check if checkpoints directory exists
    if not os.path.exists(CNN_checkpoints_DIR) or not os.listdir(CNN_checkpoints_DIR):
        print(f"Error: checkpoints directory '{CNN_checkpoints_DIR}' is empty or does not exist.")
        print("Please run 'train_target_model.py' first to generate the weight checkpoints.")
    else:
        train_diffusion_model(
            checkpoints_dir=CNN_checkpoints_DIR,
            target_model_ref_for_dims=reference_cnn.state_dict(), # Pass the state_dict
            epochs=DIFFUSION_EPOCHS,
            lr=LEARNING_RATE_DIFFUSION,
            batch_size=BATCH_SIZE_DIFFUSION,
            time_emb_dim=TIME_EMB_DIM_DIFF,
            hidden_dim_diff_model=HIDDEN_DIM_DIFF,
            save_path=SAVED_DIFFUSION_MODEL_PATH,
            max_traj_len_for_training=MAX_checkpoints_LENGTH_FOR_TRAINING
        )
