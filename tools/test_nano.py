import sys
import os

# Asegurarse de que se puede importar depth_anything
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from depth_anything.dpt import DepthAnything

def test_nano_model():
    print("Iniciando prueba del modelo Nano (vitn)...")
    
    config = {
        'encoder': 'vitn',
        # Los parametros de features y out_channels se configuran solos en dpt.py
        'use_bn': False,
        'use_clstoken': False,
        'localhub': True,
        'act_layer': torch.nn.ReLU
    }
    
    try:
        model = DepthAnything(config)
        print("\n✅ ¡Modelo instanciado correctamente con la configuración 'vitn' y ReLU!")
        
        # Calcular numero de parametros
        total_params = sum(p.numel() for p in model.parameters())
        encoder_params = sum(p.numel() for p in model.pretrained.parameters())
        decoder_params = sum(p.numel() for p in model.depth_head.parameters())
        
        print(f"\nResumen de Parámetros:")
        print(f" - Encoder (vit_nano): {encoder_params:,}")
        print(f" - Decoder (DPTHead): {decoder_params:,}")
        print(f" - Total: {total_params:,}")
        
    except Exception as e:
        print(f"\n❌ Se produjo un error al instanciar el modelo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_nano_model()
