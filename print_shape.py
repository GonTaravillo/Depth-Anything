import sys, os, torch
sys.path.insert(0, os.path.abspath('.'))
from depth_anything.dpt import DepthAnything
config = {'encoder': 'vitn', 'use_bn': False, 'use_clstoken': False, 'localhub': True, 'act_layer': torch.nn.ReLU}
model = DepthAnything(config)
print("pos_embed shape:", model.pretrained.pos_embed.shape)
print("patch_size:", model.pretrained.patch_embed.patch_size)
print("img_size:", model.pretrained.patch_embed.img_size)
