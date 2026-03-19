import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from depth_anything.dpt import DepthAnything

def compare_models():
    device = "cpu"
    
    # 1. Configuración Original (GELU) - Como el Teacher
    print("--- Verificando Modelo Original (GELU) ---")
    orig_config = {
        'encoder': 'vits',
        'use_bn': False,
        'use_clstoken': False,
        'localhub': True,
        'act_layer': torch.nn.GELU
    }
    model_gelu = DepthAnything(orig_config).to(device)
    
    # Verificar activaciones finales en GELU
    # En GELU, output_conv2[3] deberia ser GELU (no Identity)
    final_act_gelu = model_gelu.depth_head.scratch.output_conv2[3]
    print(f"Activación final en GELU: {type(final_act_gelu).__name__}")
    
    # 2. Configuración Nueva (ReLU) - Como el Student Nano
    print("\n--- Verificando Modelo Nuevo (ReLU) ---")
    new_config = {
        'encoder': 'vitn',
        'use_bn': False,
        'use_clstoken': False,
        'localhub': True,
        'act_layer': torch.nn.ReLU
    }
    model_relu = DepthAnything(new_config).to(device)
    
    # Verificar activaciones finales en ReLU
    # En ReLU, output_conv2[3] deberia ser Identity
    final_act_relu = model_relu.depth_head.scratch.output_conv2[3]
    print(f"Activación final en ReLU: {type(final_act_relu).__name__}")
    
    # 3. Prueba de salida
    dummy_input = torch.randn(1, 3, 518, 518)
    with torch.no_grad():
        out_gelu = model_gelu(dummy_input)
        out_relu = model_relu(dummy_input)
        
    print(f"\nStats Salida GELU (Original): Min {out_gelu.min().item():.4f}, Max {out_gelu.max().item():.4f}")
    print(f"Stats Salida ReLU (Nano): Min {out_relu.min().item():.4f}, Max {out_relu.max().item():.4f}")

if __name__ == "__main__":
    compare_models()
