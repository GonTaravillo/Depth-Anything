import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
from depth_anything.dpt import DepthAnything

def export_onnx():
    device = torch.device('cpu')
    
    student_config = {
        'encoder': 'vitn',
        'use_bn': False,
        'use_clstoken': False,
        'localhub': True,
        'act_layer': torch.nn.ReLU
    }
    
    model = DepthAnything(student_config)
    
    checkpoint = '/home/gonzalo/Documents/PFG/Depth-Anything_Con_version_nano_preliminar/checkpoints/depth_anything_nano_epoch_10.pth'
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Loaded {checkpoint}")
    else:
        print(f"Checkpoint not found: {checkpoint}")
        return
        
    model.eval()
    
    # Dummy input for 168x168
    dummy_input = torch.randn(1, 3, 168, 168)
    
    output_path = 'depth_anything_nano_epoch10_168x168_clean.onnx'
    
    print(f"Exporting ONNX to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None  # Force static shape!
    )
    print("Done!")

if __name__ == '__main__':
    export_onnx()
