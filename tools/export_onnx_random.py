import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from depth_anything.dpt import DepthAnything

def export_to_onnx(output_path):
    config = {
        'encoder': 'vitp',
        'use_bn': False,
        'use_clstoken': False,
        'localhub': True,
        'act_layer': torch.nn.ReLU
    }
    model = DepthAnything(config)
    model.eval()
    
    # Dummy input format, C, H, W
    dummy_input = torch.randn(1, 3, 192, 192)
    
    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        export_params=True, 
        opset_version=14,  # Used 14 recently for better compatibility with shapes sometimes, maybe 12
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output']
    )
    print("Export successful!")

if __name__ == "__main__":
    output_onnx = "depth_anything_nano_random.onnx"
    export_to_onnx(output_onnx)
