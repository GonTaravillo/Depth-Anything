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

dummy_input = torch.randn(1, 3, 112, 112)

# Hook into DPTHead
def hook(module, input, output):
    print("DPTHead output shape:", output.shape)
model.depth_head.register_forward_hook(hook)

out = model(dummy_input)
print("Final out shape:", out.shape)
