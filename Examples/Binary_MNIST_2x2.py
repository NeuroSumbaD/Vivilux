from sfp_board_config_6x6 import netlist, fpga

import __main__
import os
from time import sleep
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
            print(f"Processing pixel at {x_pos}, {y_pos}")
            data_in = image[0, x_pos: x_pos+2, y_pos:y_pos+2]
            data_out[x_pos, y_pos, :] = torch.from_numpy(mzi.applyTo(data_in.flatten()))
    
    return net.forward(data_out)

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
    )

    # Define the laser array for the MZI input
    inputLaser = SFPLaserArray(
        size=4,
        control_nets=["laser_0", "laser_1", "laser_2", "laser_3"],
        detectors=inputDetectors,
        netlist=netlist,
        board=fpga,
        # use_vibrations=False,
        # pause=1,
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
    sleep(1)  # Allow time for the voltages to settle

    print("Defining MZI (skipping initialization measurement).")
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
        initialize=False,
        check_stop=200, # set to a larger number to avoid stopping early
        skip_zeros=False, # don't skip zero vectors (gives reasonable noise output)
        one_hot=False, # send input vector in one shot (can be problematic for unbalanced lasers)
        
        # default generator is a uniform distribution on all parameters
        # step_generator=gen_from_one_hot, # use one-hot step vectors (trivial basis function for stepVectors)
        # step_generator=partial(gen_from_sparse_permutation, numHot=3), # use sparse permutation basis for stepVectors
    )
    
    print("Geeting first image...")
    img, label = dataset[0]
    print("Attempting to process a single image.")
    output = single_forward_pass(img, mzi, net)
    output_class = torch.argmax(output)
    print(f'Single inference random initialization yields '
          f'output={output_class} for label={label}')