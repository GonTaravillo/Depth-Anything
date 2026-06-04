import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
import glob

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import NormalizeImage, PrepareForNet

def main():
    # Usamos CPU porque la cuantizacion dinamica de PyTorch esta altamente optimizada para CPU
    DEVICE = 'cpu'
    
    config = {
        'encoder': 'vitn',
        'use_bn': False,
        'use_clstoken': False,
        'localhub': True,
        'act_layer': torch.nn.ReLU
    }
    
    checkpoint_path = 'checkpoints/depth_anything_nano_112x112_epoch_10.pth'
    print(f'Cargando modelo Nano (112x112) RE-ENTRENADO original: {checkpoint_path}')
    model = DepthAnything(config).to(DEVICE)
    
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    # ¡AQUI ESTA LA MAGIA!
    # Cuantizamos dinamicamente todas las capas lineales (que son el 95% del Vision Transformer) a INT8
    print("Cuantizando el modelo a INT8 nativamente con PyTorch...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    print("Modelo cuantizado con exito. Procediendo a inferencia...")
    
    img_paths = glob.glob('data/open_images_v7/train/data/*.jpg')[:5]
    
    transform = Compose([
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    for i, img_path in enumerate(img_paths):
        raw_image = cv2.imread(img_path)
        if raw_image is None:
            continue
            
        image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (112, 112))
        
        image_float = image_resized.astype(np.float32) / 255.0
        
        image_transformed = transform({'image': image_float})['image']
        image_tensor = torch.from_numpy(image_transformed).unsqueeze(0).to(DEVICE)
        
        # Inferencia con el modelo cuantizado
        with torch.no_grad():
            depth = quantized_model(image_tensor)
        
        depth = depth[0]
        
        print(f"Imagen {i} (INT8) - Min: {depth.min().item():.4f}, Max: {depth.max().item():.4f}")
        
        diff = depth.max() - depth.min()
        if diff > 0:
            depth = (depth - depth.min()) / diff * 255.0
        else:
            depth = torch.zeros_like(depth)
            
        depth = depth.numpy().astype(np.uint8)
        
        depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        
        orig_bgr = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
        separator = np.ones((112, 10, 3), dtype=np.uint8) * 255
        
        combined = cv2.hconcat([orig_bgr, separator, depth_colored])
        combined_large = cv2.resize(combined, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        
        out_path = f'PC_Inference/result_trained_quant_native_{i}.png'
        cv2.imwrite(out_path, combined_large)
        print(f"Inferencia INT8 completada para imagen {i}. Guardado en {out_path}")

if __name__ == '__main__':
    main()
