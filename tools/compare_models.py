import argparse
import cv2
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to Nano checkpoint')
    parser.add_argument('--outdir', type=str, default='./resultados')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Small Model (vits)
    print("Cargando modelo Small (vits)...")
    model_s = DepthAnything.from_pretrained('LiheYoung/depth_anything_vits14').to(DEVICE).eval()
    
    # 2. Load Nano Model (vitn)
    print(f"Cargando modelo Nano con checkpoint: {args.checkpoint}")
    from torch import nn
    model_n = DepthAnything({
        'encoder': 'vitn',
        'features': 64,
        'out_channels': [48, 96, 192, 384],
        'use_bn': False,
        'use_clstoken': False,
        'act_layer': nn.ReLU
    }).to(DEVICE).eval()
    
    model_n.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('.txt'):
            with open(args.img_path, 'r') as f:
                filenames = [line.strip() for line in f if line.strip()]
        else:
            filenames = [args.img_path]
    else:
        filenames = os.listdir(args.img_path)
        filenames = [os.path.join(args.img_path, filename) for filename in filenames if filename.endswith(('.jpg', '.png'))]
        filenames.sort()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    margin_width = 10
    caption_height = 50
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    
    for filename in tqdm(filenames):
        raw_image = cv2.imread(filename)
        h, w = raw_image.shape[:2]
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            depth_s = model_s(image)
            depth_n = model_n(image)
        
        # Interpolate and normalize Small
        depth_s = F.interpolate(depth_s[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth_s = (depth_s - depth_s.min()) / (depth_s.max() - depth_s.min()) * 255.0
        depth_s = depth_s.cpu().numpy().astype(np.uint8)
        depth_s = cv2.applyColorMap(depth_s, cv2.COLORMAP_INFERNO)
        
        # Interpolate and normalize Nano
        depth_n = F.interpolate(depth_n[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth_n = (depth_n - depth_n.min()) / (depth_n.max() - depth_n.min()) * 255.0
        depth_n = depth_n.cpu().numpy().astype(np.uint8)
        depth_n = cv2.applyColorMap(depth_n, cv2.COLORMAP_INFERNO)
        
        # Create comparison visualization
        # Layout: [Raw] [Small] [Nano]
        divider = np.ones((h, margin_width, 3), dtype=np.uint8) * 255
        combined = cv2.hconcat([raw_image, divider, depth_s, divider, depth_n])
        
        # Add labels
        canvas = np.ones((caption_height, combined.shape[1], 3), dtype=np.uint8) * 255
        labels = ['Original', 'Depth Anything (Small)', 'Depth Anything (Nano)']
        
        for i, label in enumerate(labels):
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            # Center text in each segment (w + margin)
            center_x = (w + margin_width) * i + w // 2 - text_size[0] // 2
            cv2.putText(canvas, label, (center_x, 35), font, font_scale, (0, 0, 0), font_thickness)
        
        final = cv2.vconcat([canvas, combined])
        
        base_name = os.path.basename(filename).split('.')[0]
        output_path = os.path.join(args.outdir, f'compare_{base_name}.png')
        cv2.imwrite(output_path, final)
        print(f"Resultado guardado en: {output_path}")
