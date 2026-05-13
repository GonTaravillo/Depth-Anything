import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from depth_anything.dpt import DepthAnything

config = {
    'encoder': 'vitn',
    'use_bn': False,
    'use_clstoken': False,
    'localhub': True,
    'act_layer': torch.nn.ReLU
}
model = DepthAnything(config)
model.eval()

dummy_input = torch.randn(1, 3, 112, 112)
output = model(dummy_input)
print("Output shape:", output.shape)
