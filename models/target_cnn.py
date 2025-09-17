import torch
import torch.nn as nn
import torch.nn.functional as F

class TargetCNN(nn.Module):
    def __init__(self):
        super(TargetCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flattened_size = 8 * 7 * 7

        self.fc1 = nn.Linear(self.flattened_size, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, self.flattened_size) # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Output raw logits
        return x

if __name__ == '__main__':
    # Test the model with a dummy input
    model = TargetCNN()
    print("TargetCNN model initialized.")
    print(model)

    # MNIST images are 1x28x28 (channel, height, width)
    dummy_input = torch.randn(1, 1, 28, 28) # Batch size of 1
    output = model(dummy_input)
    print(f"\nDummy input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Should be (1, 10)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")
