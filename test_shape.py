import torch
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from depth_anything.dpt import DepthAnything

config = {'encoder': 'vitn', 'use_bn': False, 'use_clstoken': False, 'localhub': True, 'act_layer': torch.nn.ReLU}
model = DepthAnything(config)

x = torch.zeros(1, 3, 112, 112)

print("Eval mode:")
model.eval()
try:
    out = model(x)
    print("Eval OK, shape:", out.shape)
except Exception as e:
    print("Eval failed:", e)

print("Train mode:")
model.train()
try:
    out = model(x)
    print("Train OK, shape:", out.shape)
except Exception as e:
    print("Train failed:", e)
