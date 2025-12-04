from sfp_board_config_6x6 import netlist, fpga

import __main__
import os
from time import time, sleep
import json

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from vivilux.hardware.detectors import DetectorArray
from vivilux.hardware.arbitrary_mzi import HardMZI_v3
from vivilux.hardware.lasers import SFPLaserArray, SFPDetectorArray


threshold_value = 0.5
binaryTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > threshold_value).float())
]
)

dataset = datasets.MNIST(
    root = "./data",
    train = True,
    download = True,
    transform = binaryTransform
)

class SimpleNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gain = nn.Parameter(torch.rand(1))
        self.bias1 = nn.Parameter(torch.randn(1,1,5))
        self.linear = nn.Linear(3380,10)
        
    def forward(self, x):
        x = x * self.gain
        x = x - self.bias1
        x = F.relu(x)
        x = self.linear(x.flatten())
        
        return x

net = SimpleNet()

def single_forward_pass(image: torch.Tensor, mzi: HardMZI_v3, net: nn.Module):
    image = image.numpy()
    data_out = torch.zeros(26,26, 5)
    for x_pos in range(26):
        for y_pos in range(26):
            # print(f"Processing pixel at {x_pos}, {y_pos}")
            data_in = image[0, x_pos: x_pos+2, y_pos:y_pos+2]
            data_out[x_pos, y_pos, :] = torch.from_numpy(mzi.applyTo(data_in.flatten()))
    
    return net.forward(data_out)

def batch_forward_pass(images: torch.Tensor, labels: torch.Tensor, mzi: HardMZI_v3, net: nn.Module):
    """Process a batch of images through the MZI and network.
    
    Args:
        images: Batch of images [batch_size, 1, 28, 28]
        labels: Batch of labels [batch_size]
        mzi: Hardware MZI instance
        net: PyTorch neural network
        
    Returns:
        outputs: Network predictions [batch_size, 10]
        mzi_outputs: Raw MZI outputs for gradient calculation [batch_size, 26, 26, 5]
    """
    batch_size = images.shape[0]
    mzi_outputs = torch.zeros(batch_size, 26, 26, 5)
    
    for batch_idx in range(batch_size):
        image = images[batch_idx].numpy()
        for x_pos in range(26):
            for y_pos in range(26):
                data_in = image[0, x_pos: x_pos+2, y_pos:y_pos+2]
                mzi_outputs[batch_idx, x_pos, y_pos, :] = torch.from_numpy(
                    mzi.applyTo(data_in.flatten())
                )
    
    # Forward through the network
    outputs = torch.zeros(batch_size, 10)
    for batch_idx in range(batch_size):
        outputs[batch_idx] = net.forward(mzi_outputs[batch_idx])
    
    return outputs, mzi_outputs

def train_loop(mzi: HardMZI_v3, net: nn.Module, dataset: datasets.MNIST, 
               num_epochs: int = 50, batch_size: int = 8, learning_rate: float = 0.001,
               mzi_learning_rate: float = 0.01):
    """Training loop for MZI-based convolution with PyTorch network.
    
    Args:
        mzi: Hardware MZI instance performing convolution
        net: PyTorch neural network
        dataset: MNIST dataset
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for PyTorch optimizer
        mzi_learning_rate: Learning rate for MZI updates
    """
    net.train()
    
    # Setup optimizer and loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    for epoch in range(num_epochs):
        epoch_start = time()
        epoch_loss = 0.0
        num_batches = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            print(f"Processing batch {batch_idx+1}/{len(dataloader)}")
            # Forward pass through MZI and network
            # Note: MZI outputs need to track gradients for backprop
            mzi_outputs_list = []
            outputs_list = []
            
            for img_idx in range(images.shape[0]):
                print(f"  Processing image {img_idx+1}/{images.shape[0]}")
                image_start = time()
                image = images[img_idx].numpy()
                mzi_output = torch.zeros(26, 26, 5, requires_grad=False)
                
                # Store input patches for gradient calculation
                input_patches = []
                for x_pos in range(26):
                    for y_pos in range(26):
                        data_in = image[0, x_pos: x_pos+2, y_pos:y_pos+2]
                        input_patches.append(data_in.flatten())
                        mzi_output[x_pos, y_pos, :] = torch.from_numpy(
                            mzi.applyTo(data_in.flatten())
                        )
                
                mzi_outputs_list.append((mzi_output, input_patches))
                outputs_list.append(net.forward(mzi_output))
                image_time = time() - image_start
                print(f"\tImage processed in {image_time:.2f} seconds, {image_time/676:.4f} seconds per patch")
            
            outputs = torch.stack(outputs_list)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            num_batches += 1
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Backward pass for PyTorch parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # ========================================================================
            # Calculate gradient for MZI convolution kernels
            # ========================================================================
            # The MZI implements 5 flattened 2x2 convolution kernels as a (5,4) matrix.
            # To update the kernels, we need to:
            # 1. Get gradient w.r.t. each MZI output position (dL/dy)
            # 2. Multiply by the input patch at that position (x)
            # 3. This gives us dL/dW = dL/dy * x (chain rule for matrix multiply)
            # 4. Accumulate across all spatial positions and batch
            # ========================================================================
            
            mzi_gradient = torch.zeros(5, 4)  # 5 kernels, each 2x2 (4 elements)
            
            for img_idx in range(images.shape[0]):
                mzi_output, input_patches = mzi_outputs_list[img_idx]
                
                # Get gradient w.r.t. MZI output from the network
                mzi_output.requires_grad_(True)
                output = net.forward(mzi_output)
                output_loss = criterion(output.unsqueeze(0), labels[img_idx].unsqueeze(0))
                output_loss.backward()
                
                if mzi_output.grad is not None:
                    # Aggregate gradients for each kernel
                    # Each position in the output corresponds to a 2x2 input patch
                    patch_idx = 0
                    for x_pos in range(26):
                        for y_pos in range(26):
                            grad_output = mzi_output.grad[x_pos, y_pos, :]  # [5]
                            input_patch = torch.from_numpy(input_patches[patch_idx])  # [4]
                            
                            # Gradient is outer product of input and output gradient
                            # For each kernel: dL/dW_k = sum over positions of (dL/dy_k * x)
                            for kernel_idx in range(5):
                                mzi_gradient[kernel_idx] += (
                                    grad_output[kernel_idx] * input_patch
                                )
                            patch_idx += 1
            
            # Average gradient over batch
            mzi_gradient = mzi_gradient.numpy() / images.shape[0]
            
            # Apply gradient update to MZI using hardware's ApplyDelta method
            # Negative gradient for gradient descent
            mzi_delta = -mzi_learning_rate * mzi_gradient
            print(f"Expecting to update MZI with delta:\n{mzi_delta}")
            
            # Apply the update to the hardware MZI
            try:
                mzi.ApplyDelta(
                    mzi_delta, 
                    eta=1.0,  # Step size (already included in mzi_learning_rate)
                    numDirections=6,  # Number of gradient directions to sample
                    numSteps=20,  # Number of optimization steps
                    earlyStop=1e-2,
                    verbose=False
                )
            except Exception as e:
                print(f"Warning: MZI update failed: {e}")
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}")
        
        # Print epoch statistics
        avg_loss = epoch_loss / num_batches
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Complete - "
              f"Avg Loss: {avg_loss:.4f},\tAccuracy: {accuracy:.2f}%,\tTime: {time() - epoch_start:.2f} seconds")
    

print("Entering netlist context...")
# Define experiment within netlist context
with netlist:
    # Define the detector arrays before and after the MZI
    inputDetectors = DetectorArray(
        size=4,
        nets=["PD_2_0", "PD_3_0",  "PD_4_0", "PD_5_0",],# "PD_5_0", "PD_6_0",],
        netlist=netlist,
        transimpedance=220e3,  # 220k ohms (TODO: double-check if these detectors are 220k or 10k)
    )
    raw_outputDetectors = DetectorArray(
        size=5,
        nets=["PD_1_5","PD_2_5", "PD_3_5",  "PD_4_5", "PD_5_5",],# "PD_5_5", "PD_6_5",],
        netlist=netlist,
        transimpedance=220e3,  # 220k ohms
        min_zero=False, # Allow negative readings (negative possible, but unlikely, when using SFP lasers with vibrations instead of full on/off)
    )

    # Define the laser array for the MZI input
    inputLaser = SFPLaserArray(
        size=4,
        control_nets=["laser_0", "laser_1", "laser_2", "laser_3"],
        detectors=inputDetectors,
        netlist=netlist,
        board=fpga,
        # use_vibrations=False,
        # pause=50e-3,
    )

    outputDetectors = SFPDetectorArray(
        detectors=raw_outputDetectors,
        lasers=inputLaser,  # Use the input laser for calibration
    )
    
    # Load bar state MZI configuration and set initial voltages
    # try:
    #     # load bar state from test folder in parent directory
    #     tests_dir = os.path.join(os.path.dirname(__main__.__file__), "tests")
    #     json_path = os.path.join(tests_dir, "5x5_bar_state_voltages.json")
    #     json_dict = json.load(open(json_path, "r"))
    #     for net, voltage in json_dict.items():
    #         netlist[net].vout(voltage)
    # except FileNotFoundError:
    #     raise RuntimeError("Bar state voltages file not found, cannot proceed with calibration.")
    
    print("Turning the lasers off (vibration).")
    inputLaser.setNormalized([0, 0, 0, 0])  # Set initial laser powers to 0

    print("Defining MZI...")
    # initialize the MZI with the defined components
    mzi = HardMZI_v3(
        shape=(5, 4),
        outputDetectors=outputDetectors,
        inputLaser=inputLaser,
        psPins=["3_1_i", "2_2_i", "4_2_i", "3_3_i", "2_4_i", "4_4_i", # main pins for 4x4 subset
                "2_2_o", "4_2_o", "3_3_o", "2_4_o", "4_4_o", # PHI phase shifters
                "1_1_i", "1_3_o", "1_3_i", "1_5_i", "1_5_o",
                ],
        netlist=netlist,
        updateMagnitude = 0.8,
        updateMagDecay = 0.985,
        # ps_delay=50e-3,  # delay for phase shifter voltage to settle
        num_samples=1,
        # initialize=False,
        check_stop=200, # set to a larger number to avoid stopping early
        skip_zeros=False, # don't skip zero vectors (gives reasonable noise output)
        one_hot=False, # send input vector in one shot (can be problematic for unbalanced lasers)
        
        # default generator is a uniform distribution on all parameters
        # step_generator=gen_from_one_hot, # use one-hot step vectors (trivial basis function for stepVectors)
        # step_generator=partial(gen_from_sparse_permutation, numHot=3), # use sparse permutation basis for stepVectors
    )
    
    # print("Geting first image...")
    # img, label = dataset[0]
    # print("Attempting to process a single image.")
    # output = single_forward_pass(img, mzi, net)
    # output_class = torch.argmax(output)
    # print(f'Single inference random initialization yields '
    #       f'output={output_class} for label={label}')
    
    
    # Train model and MZI
    print("Starting training loop...")
    
    # Create a small subset of the dataset for initial testing (16 images)
    small_dataset = torch.utils.data.Subset(dataset, indices=range(16))
    print(f"Using small dataset with {len(small_dataset)} images")
    
    train_loop(
        mzi=mzi,
        net=net,
        dataset=small_dataset,  # Use the small subset instead of full dataset
        num_epochs=1,
        batch_size=4,
        learning_rate=0.001,
        mzi_learning_rate=1, # NOTE: increased learning rate to only change MZI when significant change is needed
    )
    
    print("Training complete!")
