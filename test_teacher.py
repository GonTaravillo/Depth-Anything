import torch
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from depth_anything.dpt import depth_anything_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = depth_anything_model(pretrained=True, encoder='vitl').to(device)
model.eval()

x = torch.zeros(1, 3, 518, 518).to(device)
try:
    with torch.no_grad():
        out = model(x)
    print("Teacher OK")
except Exception as e:
    print("Teacher failed:", e)
    import traceback
    traceback.print_exc()
