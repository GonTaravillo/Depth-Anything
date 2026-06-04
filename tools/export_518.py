import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from depth_anything.dpt import DepthAnything

def export_to_onnx(checkpoint_path, output_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    config = {
        'encoder': 'vitn',
        'use_bn': False,
        'use_clstoken': False,
        'localhub': True,
        'act_layer': torch.nn.ReLU
    }
    model = DepthAnything(config)
    
    if list(checkpoint.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.eval()
    
    # 518x518 is the optimal resolution for DepthAnything
    dummy_input = torch.randn(1, 3, 518, 518)
    
    model.pretrained.pos_embed.data = model.pretrained.interpolate_pos_encoding(
        torch.zeros(1, 65, 192), 518, 518
    )
    
    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        export_params=True, 
        opset_version=12, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output']
    )
    print("Export successful!")

if __name__ == "__main__":
    # Usamos el checkpoint original (no el re-entrenado para 112x112)
    checkpoint = "checkpoints/depth_anything_nano_epoch_10.pth"
    output_onnx = "depth_anything_nano_518x518.onnx"
    export_to_onnx(checkpoint, output_onnx)
