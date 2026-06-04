import torch
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from depth_anything.dpt import DepthAnything

device = 'cuda'
config = {'encoder': 'vitn', 'use_bn': False, 'use_clstoken': False, 'localhub': True, 'act_layer': torch.nn.ReLU}
model = DepthAnything(config).to(device)

x = torch.zeros(4, 3, 112, 112).to(device)

model.train()
try:
    with torch.cuda.amp.autocast(enabled=True):
        out = model(x)
    print("Train AMP OK, shape:", out.shape)
except Exception as e:
    print("Train AMP failed:", e)
