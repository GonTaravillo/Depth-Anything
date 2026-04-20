import os
import sys
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import glob
import argparse

def get_transform(target_size=192):
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def extract_to_bin(image_path, output_path, target_size=192):
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    transform = get_transform(target_size=target_size)
    tensor = transform(image) # [3, 192, 192] shape float tensor
    
    # We flatten the tensor and write raw 32-bit floats
    float_array = tensor.numpy().flatten().astype(np.float32)
    
    with open(output_path, 'wb') as f:
        f.write(float_array.tobytes())
        
    print(f"Exported to {output_path} ({len(float_array) * 4} bytes)")

def main():
    parser = argparse.ArgumentParser(description="Convert JPG to raw FLOAT32 bin array for ESP32 SD Card testing.")
    parser.add_argument("--image", type=str, help="Path to input .jpg image", default=None)
    parser.add_argument("--out", type=str, help="Path to output .bin file", default="image.bin")
    
    args = parser.parse_args()
    
    if args.image is None:
        # Auto-find an image to use if none provided
        print("No image provided. Searching open_images_v7 train set...")
        base_data = os.path.join('data', 'open_images_v7', 'train', 'data')
        if not os.path.exists(base_data):
            print(f"Could not find test data directory: {base_data}")
            sys.exit(1)
            
        images = glob.glob(os.path.join(base_data, "*.jpg"))
        if len(images) == 0:
            print("No images found.")
            sys.exit(1)
            
        args.image = images[0]
        
    extract_to_bin(args.image, args.out)

if __name__ == "__main__":
    main()
