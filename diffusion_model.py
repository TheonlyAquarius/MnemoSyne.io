import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleWeightSpaceDiffusion(nn.Module):
    """
    A simplified diffusion model that operates directly in the weight space
    of another neural network (the target CNN). It predicts a slightly "denoised"
    set of weights given a "noisier" set and a timestep embedding.

    This is a conceptual model. For complex weight spaces, this direct approach
    might be challenging. U-Net architectures are common in image diffusion
    but adapting them to arbitrary weight vector spaces requires careful design.
    Here, we'll use a Multi-Layer Perceptron (MLP) for simplicity, assuming
    the flattened weights of the target model as input.
    """
    def __init__(self, target_model_flat_dim, time_emb_dim=64, hidden_dim=512):
        super(SimpleWeightSpaceDiffusion, self).__init__()
        self.target_model_flat_dim = target_model_flat_dim
        self.time_emb_dim = time_emb_dim

        # Time embedding network
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Main diffusion network (MLP based)
        # Input: flattened noisy weights + time embedding
        # Output: predicted denoised weights (or noise to subtract)
        self.diffusion_mlp = nn.Sequential(
            nn.Linear(target_model_flat_dim + time_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_model_flat_dim) # Predicts the denoised weights
        )

    def forward(self, noisy_weights_flat, t):
        """
        Args:
            noisy_weights_flat (torch.Tensor): Flattened "noisy" weights of the target model.
                                                Shape: (batch_size, target_model_flat_dim)
            t (torch.Tensor): Timestep, normalized (e.g., 0 to 1 or discrete steps).
                              Shape: (batch_size, 1) or just (batch_size)
        Returns:
            torch.Tensor: Predicted "denoised" weights (or could be noise).
                          Shape: (batch_size, target_model_flat_dim)
        """
        if t.ndim == 1:
            t = t.unsqueeze(-1) # Ensure t is [batch_size, 1] for time_mlp

        time_embedding = self.time_mlp(t.float()) # Ensure t is float

        # Concatenate noisy weights and time embedding
        x = torch.cat([noisy_weights_flat, time_embedding], dim=-1)

        # Predict the denoised state (or the noise, depending on training target)
        predicted_denoised_weights_flat = self.diffusion_mlp(x)

        return predicted_denoised_weights_flat

def get_target_model_flat_dim(target_model_state_dict):
    """Helper function to get the total number of parameters in a state_dict."""
    return sum(p.numel() for p in target_model_state_dict.values())

def flatten_state_dict(state_dict):
    """Flattens a state_dict into a single vector."""
    return torch.cat([p.flatten() for p in state_dict.values()])

def unflatten_to_state_dict(flat_params, reference_state_dict):
    """Converts a flat vector back to a state_dict, using reference_state_dict for shapes and keys."""
    new_state_dict = {}
    current_pos = 0
    for key, param_ref in reference_state_dict.items():
        num_elements = param_ref.numel()
        shape = param_ref.shape
        new_state_dict[key] = flat_params[current_pos : current_pos + num_elements].view(shape)
        current_pos += num_elements
    if current_pos != flat_params.numel():
        raise ValueError("Mismatch in number of elements during unflattening.")
    return new_state_dict


if __name__ == '__main__':
    # Example Usage (Conceptual - requires a TargetCNN model and its weights)
    from target_cnn import TargetCNN

    # 1. Get a reference state_dict and its flattened dimension
    print("Initializing a dummy TargetCNN to get weight dimensions...")
    dummy_cnn = TargetCNN()
    dummy_state_dict = dummy_cnn.state_dict()
    flat_dim = get_target_model_flat_dim(dummy_state_dict)
    print(f"Flattened dimension of TargetCNN weights: {flat_dim}")

    # 2. Initialize the diffusion model
    diffusion_model = SimpleWeightSpaceDiffusion(target_model_flat_dim=flat_dim)
    print("\nSimpleWeightSpaceDiffusion model initialized:")
    print(diffusion_model)

    # 3. Create dummy input for the diffusion model
    batch_size = 4
    # Simulate "noisy" flattened weights (e.g., from an early training stage of TargetCNN)
    dummy_noisy_weights = torch.randn(batch_size, flat_dim)
    # Simulate timesteps (e.g., representing how "noisy" these weights are)
    # Timesteps could be normalized, e.g., 0 for fully random, T_max for trained.
    # For the diffusion model's forward pass (denoising step), t would represent the current noise level.
    dummy_timesteps = torch.rand(batch_size, 1) * 10 # Example: 10 diffusion steps

    print(f"\nDummy noisy_weights_flat shape: {dummy_noisy_weights.shape}")
    print(f"Dummy timesteps shape: {dummy_timesteps.shape}")

    # 4. Pass through the diffusion model
    predicted_denoised_weights = diffusion_model(dummy_noisy_weights, dummy_timesteps)
    print(f"Predicted denoised_weights_flat shape: {predicted_denoised_weights.shape}")

    # 5. Test flattening and unflattening
    print("\nTesting state_dict flattening and unflattening...")
    original_flat_weights = flatten_state_dict(dummy_state_dict)
    print(f"Original flat weights shape: {original_flat_weights.shape}")

    reconstructed_state_dict = unflatten_to_state_dict(original_flat_weights, dummy_state_dict)

    # Verify reconstruction
    all_match = True
    for key in dummy_state_dict:
        if not torch.equal(dummy_state_dict[key], reconstructed_state_dict[key]):
            all_match = False
            print(f"Mismatch in key: {key}")
            break
    if all_match:
        print("Successfully flattened and unflattened state_dict. All tensors match.")
    else:
        print("Error in flattening/unflattening process.")

    num_params_diffusion = sum(p.numel() for p in diffusion_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in Diffusion Model: {num_params_diffusion}")
