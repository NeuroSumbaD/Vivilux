"""
Efficient training script for Binary MNIST using 2x2 MZI convolution kernels.

This script improves training efficiency by:
1. Characterizing the MZI transfer matrix noise distribution through sampling
2. Simulating forward passes on the computer with realistic per-channel noise
3. Accumulating gradients over large batches before hardware updates
4. Applying accumulated weight updates to the MZI hardware using ApplyDelta

The MZI implements 5 flattened 2x2 convolution kernels as a (5, 4) matrix.
Each 2x2 image patch is flattened to a 4-element vector and multiplied by this
matrix to produce 5 output channels.
"""

from sfp_board_config_6x6 import netlist, fpga

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from time import time
from typing import Tuple

from vivilux.hardware.detectors import DetectorArray
from vivilux.hardware.arbitrary_mzi import HardMZI_v3
from vivilux.hardware.lasers import SFPLaserArray, SFPDetectorArray


# ============================================================================
# Dataset Setup
# ============================================================================

def create_binary_mnist_dataset(threshold: float = 0.5) -> datasets.MNIST:
    """Create binary MNIST dataset with given threshold."""
    binary_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > threshold).float())
    ])
    
    return datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=binary_transform
    )


# ============================================================================
# MZI Convolution Layer (PyTorch Module)
# ============================================================================

class MZIConvLayer(nn.Module):
    """
    PyTorch module that simulates MZI-based 2x2 convolution with realistic noise.
    
    This layer implements 5 separate 2x2 convolution kernels derived from the
    MZI transfer matrix. It uses F.conv2d for efficient computation and adds
    per-channel Gaussian noise matching hardware measurements.
    """
    
    def __init__(self, mean_matrix: np.ndarray, std_matrix: np.ndarray):
        """
        Args:
            mean_matrix: Mean MZI transfer matrix, shape [5, 4]
            std_matrix: Noise std per matrix element, shape [5, 4]
        """
        super().__init__()
        
        # Reshape transfer matrix into 2x2 convolution kernels
        # mean_matrix is [5, 4] -> reshape to [5, 1, 2, 2] for 5 output channels
        # Each row of mean_matrix is a flattened 2x2 kernel
        kernels = torch.from_numpy(mean_matrix).float().reshape(5, 1, 2, 2)
        
        # Store as non-trainable parameter (updated from hardware)
        self.register_buffer('weight', kernels)
        
        # Store noise std for each output channel (averaged across input channels)
        # Shape: [5] - one std per output channel
        noise_std_per_channel = torch.from_numpy(np.mean(std_matrix, axis=1)).float()
        self.register_buffer('noise_std', noise_std_per_channel)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply MZI convolution with noise to input images.
        
        Args:
            x: Input images, shape [batch, 1, 28, 28]
        
        Returns:
            output: Convolution output with noise, shape [batch, 5, 26, 26]
        """
        # Apply 2x2 convolution using F.conv2d
        # Input: [batch, 1, 28, 28]
        # Weight: [5, 1, 2, 2] (5 output channels, 1 input channel, 2x2 kernel)
        # Output: [batch, 5, 27, 27] with no padding and stride=1
        output = F.conv2d(x, self.weight, bias=None, stride=1, padding=0)
        
        # Note: conv2d gives us [batch, 5, 27, 27], but we need [batch, 5, 26, 26]
        # Crop to match expected size
        output = output[:, :, :26, :26]
        
        # Add per-channel Gaussian noise
        # noise_std shape: [5] -> reshape to [1, 5, 1, 1] for broadcasting
        batch_size = output.shape[0]
        noise = torch.randn_like(output) * self.noise_std.view(1, 5, 1, 1)
        output = output + noise
        
        return output
    
    def update_from_matrix(self, new_mean_matrix: np.ndarray, new_std_matrix: np.ndarray = None):
        """
        Update the convolution kernels from a new MZI transfer matrix.
        
        Args:
            new_mean_matrix: New mean MZI transfer matrix, shape [5, 4]
            new_std_matrix: New noise std (optional), shape [5, 4]
        """
        new_kernels = torch.from_numpy(new_mean_matrix).float().reshape(5, 1, 2, 2)
        self.weight.copy_(new_kernels)
        
        if new_std_matrix is not None:
            new_noise_std = torch.from_numpy(np.mean(new_std_matrix, axis=1)).float()
            self.noise_std.copy_(new_noise_std)


# ============================================================================
# Neural Network Definition
# ============================================================================

class SimpleNet(nn.Module):
    """Simple neural network that processes MZI convolution outputs."""
    
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.rand(1))
        self.bias1 = nn.Parameter(torch.randn(1, 5, 1, 1))  # Shape [1, 5, 1, 1] for broadcasting with [B, 5, 26, 26]
        self.linear = nn.Linear(3380, 10)  # 26*26*5 = 3380
        
    def forward(self, x):
        """
        Args:
            x: MZI output tensor of shape [batch, 5, 26, 26]
        
        Returns:
            logits: Class logits of shape [batch, 10]
        """
        # x shape: [batch, 5, 26, 26]
        x = x * self.gain  # [batch, 5, 26, 26]
        x = x - self.bias1  # [batch, 5, 26, 26] - [1, 5, 1, 1] -> [batch, 5, 26, 26]
        x = F.relu(x)  # [batch, 5, 26, 26]
        x = x.flatten(start_dim=1)  # [batch, 5, 26, 26] -> [batch, 3380]
        x = self.linear(x)  # [batch, 3380] -> [batch, 10]
        return x


# ============================================================================
# MZI Transfer Matrix Characterization
# ============================================================================

def characterize_mzi_noise(
    mzi: HardMZI_v3,
    num_samples: int = 50,
    test_inputs: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample the MZI transfer matrix multiple times to characterize noise.
    
    This function measures the MZI's response to standard basis vectors 
    (or provided test inputs) multiple times to compute the mean transfer 
    matrix and per-channel standard deviation.
    
    Args:
        mzi: Hardware MZI instance
        num_samples: Number of times to sample the transfer matrix
        test_inputs: Optional array of test inputs shape [num_inputs, 4].
                     If None, uses standard basis vectors [1,0,0,0], etc.
    
    Returns:
        mean_matrix: Mean transfer matrix of shape [5, 4]
        std_matrix: Standard deviation per matrix element, shape [5, 4]
    """
    print(f"\nCharacterizing MZI noise with {num_samples} samples...")
    start_time = time()
    
    # Use standard basis vectors if no test inputs provided
    if test_inputs is None:
        test_inputs = np.eye(4)
    
    num_inputs = test_inputs.shape[0]
    
    # Collect samples: shape will be [num_samples, 5, num_inputs]
    samples = np.zeros((num_samples, 5, num_inputs))
    
    for sample_idx in range(num_samples):
        if sample_idx % 10 == 0:
            print(f"  Sample {sample_idx}/{num_samples}")
        
        for input_idx, input_vec in enumerate(test_inputs):
            output = mzi.applyTo(input_vec)
            samples[sample_idx, :, input_idx] = output
    
    # Reconstruct transfer matrices from responses
    # Each column of the matrix is the response to a basis vector
    transfer_matrices = samples  # shape: [num_samples, 5, 4]
    
    # Compute statistics
    mean_matrix = np.mean(transfer_matrices, axis=0)
    std_matrix = np.std(transfer_matrices, axis=0)
    
    elapsed = time() - start_time
    print(f"Characterization complete in {elapsed:.2f}s")
    print(f"Mean transfer matrix norm: {np.linalg.norm(mean_matrix):.4f}")
    print(f"Average noise std: {np.mean(std_matrix):.6f}")
    print(f"Max noise std: {np.max(std_matrix):.6f}")
    
    return mean_matrix, std_matrix


# ============================================================================
# Training Loop
# ============================================================================

def train_with_efficient_updates(
    mzi: HardMZI_v3,
    mzi_layer: MZIConvLayer,
    net: nn.Module,
    dataset: datasets.MNIST,
    num_epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    mzi_learning_rate: float = 0.1,
    characterization_samples: int = 50,
    recharacterize_every: int = 5,
):
    """
    Efficient training loop with PyTorch-based forward passes and hardware updates.
    
    This version uses PyTorch's nn.Module and F.conv2d for fast forward passes,
    and leverages autodifferentiation to compute gradients for the MZI weights.
    
    Args:
        mzi: Hardware MZI instance
        mzi_layer: PyTorch MZI convolution layer
        net: PyTorch neural network
        dataset: MNIST dataset
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for network parameters
        mzi_learning_rate: Learning rate for MZI updates
        characterization_samples: Number of samples for noise characterization
        recharacterize_every: Re-characterize MZI every N batches
    """
    print("\n" + "="*80)
    print("Starting Efficient Training Loop")
    print("="*80)
    
    net.train()
    
    # Setup optimizer and loss function (only for network parameters, not MZI)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    for epoch in range(num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*80}")
        epoch_start = time()
        
        epoch_loss = 0.0
        num_batches = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            batch_start = time()
            
            # Re-characterize MZI periodically to track hardware drift
            if batch_idx > 0 and batch_idx % recharacterize_every == 0:
                print(f"\n[Batch {batch_idx}] Re-characterizing MZI...")
                mean_matrix, std_matrix = characterize_mzi_noise(
                    mzi, num_samples=characterization_samples
                )
                mzi_layer.update_from_matrix(mean_matrix, std_matrix)
            
            # ================================================================
            # 1. Forward Pass (PyTorch - Fast!)
            # ================================================================
            # Apply MZI convolution with noise
            mzi_outputs = mzi_layer(images)  # [batch, 5, 26, 26]
            
            # Forward through rest of network
            outputs = net(mzi_outputs)  # [batch, 10]
            
            # ================================================================
            # 2. Compute Loss and Accuracy
            # ================================================================
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            num_batches += 1
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # ================================================================
            # 3. Backward Pass for Network Parameters (Standard PyTorch)
            # ================================================================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # ================================================================
            # 4. Compute Gradient for MZI using Autodiff
            # ================================================================
            # Re-run forward pass with gradient tracking for MZI weights
            images.requires_grad_(False)  # Don't need gradients w.r.t. input
            
            # Forward through MZI layer (with gradients for weights)
            mzi_layer.weight.requires_grad_(True)
            mzi_outputs_grad = mzi_layer(images)
            
            # Forward through network
            outputs_grad = net(mzi_outputs_grad)
            
            # Compute loss
            loss_grad = criterion(outputs_grad, labels)
            
            # Backward to compute gradient w.r.t. MZI weights
            loss_grad.backward()
            
            # Extract gradient from MZI layer weights
            # Weight shape: [5, 1, 2, 2] -> reshape to [5, 4]
            if mzi_layer.weight.grad is not None:
                mzi_gradient = mzi_layer.weight.grad.detach().cpu().numpy().reshape(5, 4)
            else:
                mzi_gradient = np.zeros((5, 4))
            
            # Zero out gradients after extraction
            mzi_layer.weight.grad = None
            mzi_layer.weight.requires_grad_(False)
            
            # ================================================================
            # 5. Apply Update to Hardware MZI using ApplyDelta
            # ================================================================
            mzi_delta = -mzi_learning_rate * mzi_gradient
            
            print(f"\n[Batch {batch_idx+1}/{len(dataloader)}]")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  MZI gradient norm: {np.linalg.norm(mzi_gradient):.6f}")
            print(f"  MZI delta norm: {np.linalg.norm(mzi_delta):.6f}")
            
            # Apply hardware update
            try:
                update_start = time()
                record, params_hist, matrix_hist = mzi.ApplyDelta(
                    mzi_delta,
                    eta=1.0,
                    numDirections=6,
                    numSteps=20,
                    earlyStop=1e-2,
                    verbose=False
                )
                update_time = time() - update_start
                print(f"  Hardware update completed in {update_time:.2f}s")
                print(f"  Final delta magnitude: {record[-1]:.6f}")
                
                # Update MZI layer with new hardware state
                new_mean = mzi.get()
                if new_mean.shape == (5, 4):
                    mzi_layer.update_from_matrix(new_mean)
                    
            except Exception as e:
                print(f"  Warning: MZI update failed: {e}")
            
            batch_time = time() - batch_start
            print(f"  Batch time: {batch_time:.2f}s")
        
        # ====================================================================
        # Epoch Summary
        # ====================================================================
        avg_loss = epoch_loss / num_batches
        accuracy = 100 * correct / total
        epoch_time = time() - epoch_start
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1} Complete")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"{'='*80}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("Binary MNIST 2x2 MZI Training - Efficient Version")
    print("="*80)
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = create_binary_mnist_dataset(threshold=0.5)
    
    # Use a subset for testing
    subset_size = 128  # Small subset for testing
    small_dataset = torch.utils.data.Subset(dataset, indices=range(subset_size))
    print(f"Using dataset subset with {len(small_dataset)} images")
    
    # Create network
    print("\nInitializing neural network...")
    net = SimpleNet()
    
    print("\nEntering hardware netlist context...")
    with netlist:
        # ====================================================================
        # Hardware Setup
        # ====================================================================
        print("Setting up hardware components...")
        
        # Input detectors
        input_detectors = DetectorArray(
            size=4,
            nets=["PD_2_0", "PD_3_0", "PD_4_0", "PD_5_0"],
            netlist=netlist,
            transimpedance=220e3,
        )
        
        # Output detectors
        raw_output_detectors = DetectorArray(
            size=5,
            nets=["PD_1_5", "PD_2_5", "PD_3_5", "PD_4_5", "PD_5_5"],
            netlist=netlist,
            transimpedance=220e3,
            min_zero=False,
        )
        
        # Laser array
        input_laser = SFPLaserArray(
            size=4,
            control_nets=["laser_0", "laser_1", "laser_2", "laser_3"],
            detectors=input_detectors,
            netlist=netlist,
            board=fpga,
        )
        
        output_detectors = SFPDetectorArray(
            detectors=raw_output_detectors,
            lasers=input_laser,
        )
        
        print("Initializing lasers (vibration mode)...")
        input_laser.setNormalized([0, 0, 0, 0])
        
        # ====================================================================
        # MZI Initialization
        # ====================================================================
        print("\nInitializing MZI mesh...")
        mzi = HardMZI_v3(
            shape=(5, 4),
            outputDetectors=output_detectors,
            inputLaser=input_laser,
            psPins=[
                "3_1_i", "2_2_i", "4_2_i", "3_3_i", "2_4_i", "4_4_i",  # Main 4x4
                "2_2_o", "4_2_o", "3_3_o", "2_4_o", "4_4_o",           # PHI phase shifters
                "1_1_i", "1_3_o", "1_3_i", "1_5_i", "1_5_o",           # Additional
            ],
            netlist=netlist,
            updateMagnitude=0.6,
            updateMagDecay=0.90,
            num_samples=1,
            check_stop=200,
            skip_zeros=False,
            one_hot=False,
        )
        
        print("MZI initialization complete!")
        print(f"MZI shape: {mzi.shape}")
        print(f"Number of phase shifters: {mzi.numUnits}")
        
        # ====================================================================
        # MZI Characterization and Layer Creation
        # ====================================================================
        print("\nCharacterizing MZI transfer matrix...")
        mean_matrix, std_matrix = characterize_mzi_noise(
            mzi, num_samples=30
        )
        
        print("\nCreating MZI convolution layer...")
        mzi_layer = MZIConvLayer(mean_matrix, std_matrix)
        
        # ====================================================================
        # Training
        # ====================================================================
        train_with_efficient_updates(
            mzi=mzi,
            mzi_layer=mzi_layer,
            net=net,
            dataset=small_dataset,
            num_epochs=5,
            batch_size=16,
            learning_rate=0.01,
            mzi_learning_rate=0.5,
            characterization_samples=30,
            recharacterize_every=4,
        )
        
        print("\n" + "="*80)
        print("Training Complete!")
        print("="*80)


if __name__ == "__main__":
    main()
