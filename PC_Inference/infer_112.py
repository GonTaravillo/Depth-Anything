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
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Instanciar el modelo (Alimentado por vitn y ReLU)
    config = {
        'encoder': 'vitn',
        'use_bn': False,
        'use_clstoken': False,
        'localhub': True,
        'act_layer': torch.nn.ReLU
    }
    
    checkpoint_path = 'checkpoints/depth_anything_nano_112x112_epoch_10.pth'
    print(f'Cargando modelo Nano (112x112) RE-ENTRENADO con checkpoint: {checkpoint_path}')
    model = DepthAnything(config).to(DEVICE)
    
    # Cargar pesos
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 2. Conseguir las 5 imagenes exactas
    img_paths = glob.glob('data/open_images_v7/train/data/*.jpg')[:5]
    
    # 3. Preparar el transform para 112x112 (sin usar Resize interno de 518)
    transform = Compose([
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    for i, img_path in enumerate(img_paths):
        raw_image = cv2.imread(img_path)
        if raw_image is None:
            continue
            
        # Redimensionar la imagen fisicamente a 112x112 para imitar al ESP32
        image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (112, 112))
        
        # Guardar imagen para el transform
        image_float = image_resized.astype(np.float32) / 255.0
        
        # Aplicar normalizacion
        image_transformed = transform({'image': image_float})['image']
        image_tensor = torch.from_numpy(image_transformed).unsqueeze(0).to(DEVICE)
        
        # Inferencia
        with torch.no_grad():
            depth = model(image_tensor)
        
        # Eliminar las dimensiones de batch (el modelo devuelve [B, H, W])
        depth = depth[0]
        
        print(f"Imagen {i} - Min: {depth.min().item():.4f}, Max: {depth.max().item():.4f}")
        
        # Escalar a 0-255 para visualizar
        diff = depth.max() - depth.min()
        if diff > 0:
            depth = (depth - depth.min()) / diff * 255.0
        else:
            depth = torch.zeros_like(depth)
            
        depth = depth.cpu().numpy().astype(np.uint8)
        
        # Aplicar mapa de calor
        depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        
        # Juntar la imagen original 112x112 y el mapa de profundidad
        orig_bgr = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
        
        # Añadir un separador blanco de 10 pixeles
        separator = np.ones((112, 10, 3), dtype=np.uint8) * 255
        
        combined = cv2.hconcat([orig_bgr, separator, depth_colored])
        
        # Escalar x4 para que se vea bien en el PC (112x112 es muy pequeño)
        combined_large = cv2.resize(combined, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        
        out_path = f'PC_Inference/result_trained_image_{i}.png'
        cv2.imwrite(out_path, combined_large)
        print(f"Inferencia completada para imagen {i}. Guardado en {out_path}")

if __name__ == '__main__':
    main()
