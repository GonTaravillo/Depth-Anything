import sys, os, torch
sys.path.insert(0, os.path.abspath('.'))
from depth_anything.dpt import DepthAnything

config = {
    'encoder': 'vitn',
    'use_bn': False,
    'use_clstoken': False,
    'localhub': True,
    'act_layer': torch.nn.ReLU
}

# we need to make sure dpt.py handles kwargs like patch_size
try:
    model = DepthAnything(config)
    print("Default pos embed:", model.pretrained.pos_embed.shape)
except Exception as e:
    print("Error:", e)

