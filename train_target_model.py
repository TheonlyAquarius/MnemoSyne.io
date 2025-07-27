import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import copy

from target_cnn import TargetCNN

def train_target_cnn(epochs=50, lr=0.0015, batch_size=75, save_interval_epochs=5, checkpoints_dir="checkpoints_weights"):
    """
    Trains the TargetCNN model on MNIST and saves weights at specified intervals.
    """
    print("Starting Target CNN training...")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create checkpoints directory if it doesn't exist
    os.makedirs(checkpoints_dir, exist_ok=True)
    print(f"checkpoints weights will be saved in: {checkpoints_dir}")

    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST mean and std
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print("MNIST dataset loaded.")

    # Initialize model, loss, and optimizer
    model = TargetCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Model, Criterion, Optimizer initialized.")

    # Save initial random weights (epoch 0)
    initial_weights_path = os.path.join(checkpoints_dir, f"weights_epoch_0.pth")
    torch.save(copy.deepcopy(model.state_dict()), initial_weights_path)
    print(f"Saved initial random weights to {initial_weights_path}")

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (batch_idx + 1) % 100 == 0: # Print progress every 100 batches
                print(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch}/{epochs}] completed. Average Training Loss: {epoch_loss:.4f}")

        # Save model weights at specified interval
        if epoch % save_interval_epochs == 0:
            weights_path = os.path.join(checkpoints_dir, f"weights_epoch_{epoch}.pth")
            # We save a deepcopy of the state_dict to avoid issues if the model continues training
            # while another part of the system might be reading the file.
            torch.save(copy.deepcopy(model.state_dict()), weights_path)
            print(f"Saved model weights to {weights_path}")

        # Evaluate on test set
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                test_loss += criterion(outputs, target).item()
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")

    print("Target CNN training finished.")
    final_weights_path = os.path.join(checkpoints_dir, f"weights_epoch_final.pth")
    torch.save(copy.deepcopy(model.state_dict()), final_weights_path)
    print(f"Saved final model weights to {final_weights_path}")

if __name__ == '__main__':
    # Configuration for the training run
    # For a quick test, you might reduce epochs. For actual checkpoints capture, more epochs are better.
    NUM_EPOCHS = 50 # Example: 5 epochs for a quick run, could be 20-50 for better checkpoints
    LEARNING_RATE = 0.001
    BATCH_SIZE = 75
    SAVE_INTERVAL = 5 # Save weights after every epoch
    checkpoints_SAVE_DIR = "checkpoints_weights"

    train_target_cnn(
        epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        save_interval_epochs=SAVE_INTERVAL,
        checkpoints_dir=checkpoints_SAVE_DIR
    )
