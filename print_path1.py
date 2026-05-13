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

# Let's fix dpt.py h, w dynamically
model.forward = model.forward.__get__(model, DepthAnything) # Reset just in case? No
# We'll just patch it:
def new_forward(self, x):
    h, w = x.shape[-2:]
    features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
    patch_h, patch_w = h // 14, w // 14
    depth = self.depth_head(features, patch_h, patch_w)
    return depth
import types
model.forward = types.MethodType(new_forward, model)

dummy = torch.randn(1, 3, 224, 224)
out = model(dummy)
print("Out shape for 224:", out.shape)

dummy2 = torch.randn(1, 3, 112, 112)
out2 = model(dummy2)
print("Out shape for 112:", out2.shape)
