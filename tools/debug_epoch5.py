import torch
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from depth_anything.dpt import DepthAnything

def debug_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"File not found: {checkpoint_path}")
        return

    print(f"Inspecting checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check weight statistics for some layers
    layers = [
        'pretrained.patch_embed.proj.weight',
        'depth_head.projects.0.weight',
        'depth_head.scratch.output_conv2.2.weight',
        'depth_head.scratch.output_conv2.2.bias'
    ]
    
    for layer in layers:
        if layer in checkpoint:
            w = checkpoint[label if 'label' in locals() else layer] # Fix logic error if any
            w = checkpoint[layer]
            print(f"Layer: {layer:40} | Mean: {w.mean().item():.6f} | Min: {w.min().item():.6f} | Max: {w.max().item():.6f}")
        else:
            print(f"Layer {layer} not found.")

    # Dummy inference
    config = {
        'encoder': 'vitn',
        'use_bn': False,
        'use_clstoken': False,
        'localhub': True,
        'act_layer': torch.nn.ReLU
    }
    model = DepthAnything(config)
    model.load_state_dict(checkpoint)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 518, 518)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nInference results on dummy input:")
    print(f"Mean: {output.mean().item():.6f} | Min: {output.min().item():.6f} | Max: {output.max().item():.6f}")
    
    # Check if there are ANY non-zero values
    non_zeros = (output > 0).sum().item()
    print(f"Non-zero pixels: {non_zeros} / {output.numel()}")

if __name__ == "__main__":
    debug_checkpoint("../checkpoints/depth_anything_nano_epoch_5.pth")
