"""
SimpleCNN Model for MNIST Classification

This module defines a simple Convolutional Neural Network architecture
suitable for the MNIST digit classification task. The architecture is
designed to be simple enough for efficient parallel training while
maintaining good classification accuracy.

Architecture:
    - Conv2d(1, 32, 3) -> ReLU -> MaxPool2d(2)
    - Conv2d(32, 64, 3) -> ReLU -> MaxPool2d(2)
    - Flatten -> Linear(1600, 128) -> ReLU -> Linear(128, 10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN for MNIST classification.
    
    Input: 28x28 grayscale images (batch_size, 1, 28, 28)
    Output: 10 class logits (batch_size, 10)
    
    This architecture achieves ~99% accuracy on MNIST while being
    computationally efficient for demonstrating parallel training.
    """
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # First convolutional block
        # Input: (batch, 1, 28, 28) -> Output: (batch, 32, 13, 13)
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=32, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        # Input: (batch, 32, 14, 14) -> Output: (batch, 64, 6, 6)
        self.conv2 = nn.Conv2d(
            in_channels=32, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # After conv layers: 64 * 7 * 7 = 3136 features
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output tensor of shape (batch_size, 10) containing class logits
        """
        # First conv block
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_num_params(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model() -> SimpleCNN:
    """Factory function to create a SimpleCNN model."""
    return SimpleCNN()


if __name__ == "__main__":
    # Test the model
    model = SimpleCNN()
    print(f"SimpleCNN Architecture:")
    print(model)
    print(f"\nTotal trainable parameters: {model.get_num_params():,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
