import os
import sys
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import glob
import argparse

def get_transform(target_size=112):
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def extract_to_bin(image_path, output_path, target_size=112):
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    transform = get_transform(target_size=target_size)
    tensor = transform(image) # [3, 112, 112] shape float tensor
    
    # Permute from [C, H, W] to [H, W, C] because ESP-DL uses NHWC format!
    tensor_hwc = tensor.permute(1, 2, 0)
    
    # We flatten the tensor and write raw 32-bit floats
    float_array = tensor_hwc.numpy().flatten().astype(np.float32)
    
    with open(output_path, 'wb') as f:
        f.write(float_array.tobytes())
        
    print(f"Exported to {output_path} ({len(float_array) * 4} bytes)")

def main():
    target_size = 112
    for i in range(5):
        img_path = f"rpi_deploy/orig_{i}.jpg"
        out_path = f"depth_anything_firmware/main/image{i}.bin"
        if os.path.exists(img_path):
            extract_to_bin(img_path, out_path, target_size)
        else:
            print(f"Warning: {img_path} not found!")

if __name__ == "__main__":
    main()
