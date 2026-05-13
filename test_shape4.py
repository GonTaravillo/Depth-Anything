import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Depth-Anything_Nano_Copia')))
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

dummy_input = torch.randn(1, 3, 224, 224)
out = model(dummy_input)
print("Final out shape:", out.shape)
