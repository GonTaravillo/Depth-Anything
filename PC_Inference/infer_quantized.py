import onnxruntime as ort
import cv2
import numpy as np
import os
import glob

from torchvision.transforms import Compose
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from depth_anything.util.transform import NormalizeImage, PrepareForNet

def main():
    onnx_path = 'working_quant/quantized.onnx'
    print(f"Cargando modelo cuantizado simulado: {onnx_path}")
    
    # Iniciar sesion ONNX Runtime
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # 2. Conseguir las 5 imagenes exactas
    img_paths = glob.glob('data/open_images_v7/train/data/*.jpg')[:5]
    
    transform = Compose([
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    for i, img_path in enumerate(img_paths):
        raw_image = cv2.imread(img_path)
        if raw_image is None:
            continue
            
        # Redimensionar la imagen fisicamente a 112x112
        image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (112, 112))
        
        image_float = image_resized.astype(np.float32) / 255.0
        
        # Aplicar normalizacion
        image_transformed = transform({'image': image_float})['image']
        image_tensor = np.expand_dims(image_transformed, axis=0) # (1, 3, 112, 112)
        
        # Inferencia con ONNX Runtime
        outputs = session.run(None, {input_name: image_tensor})
        depth = outputs[0]
        
        # Eliminar dimensiones de batch
        depth = depth[0]
        if len(depth.shape) == 3: # ONNX sometimes outputs [1, C, H, W]
            depth = depth[0]
            
        print(f"Imagen {i} - Min: {depth.min():.4f}, Max: {depth.max():.4f}")
        
        # Escalar a 0-255 para visualizar
        diff = depth.max() - depth.min()
        if diff > 0:
            depth = (depth - depth.min()) / diff * 255.0
        else:
            depth = np.zeros_like(depth)
            
        depth = depth.astype(np.uint8)
        
        # Aplicar mapa de calor
        depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        
        # Juntar la imagen original 112x112 y el mapa de profundidad
        orig_bgr = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
        separator = np.ones((112, 10, 3), dtype=np.uint8) * 255
        
        combined = cv2.hconcat([orig_bgr, separator, depth_colored])
        
        # Escalar x4 para que se vea bien en el PC
        combined_large = cv2.resize(combined, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        
        out_path = f'PC_Inference/result_quant_image_{i}.png'
        cv2.imwrite(out_path, combined_large)
        print(f"Inferencia completada para imagen {i}. Guardado en {out_path}")

if __name__ == '__main__':
    main()
