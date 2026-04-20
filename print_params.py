import sys, os, torch
sys.path.insert(0, os.path.abspath('.'))
from depth_anything.dpt import DepthAnything

config = {
    'encoder': 'vitp',
    'use_bn': False,
    'use_clstoken': False,
    'localhub': True,
    'act_layer': torch.nn.ReLU
}

model = DepthAnything(config)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n--- Depth-Anything PICO Parameters: {total_params:,} ---")
