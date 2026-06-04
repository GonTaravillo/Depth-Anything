import numpy as np
import cv2
import os

import glob
img_paths = glob.glob('data/open_images_v7/train/data/*.jpg')[:5]

for i, img_path in enumerate(img_paths):
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (112, 112))
    
    # Save the original 112x112 JPEG for comparison
    orig_jpg_path = f'orig_{i}.jpg'
    cv2.imwrite(orig_jpg_path, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
    
    # Process for the model
    image_float = image_resized.astype(np.float32) / 255.0
    
    # Normalize like in training
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_norm = (image_float - mean) / std
    
    # HWC to CHW
    image_chw = np.transpose(image_norm, (2, 0, 1))
    
    # Write binary
    bin_path = f'depth_anything_firmware/main/image{i}.bin'
    with open(bin_path, 'wb') as f:
        f.write(image_chw.tobytes())
    
    print(f"Written {image_chw.nbytes} bytes to {bin_path} and saved {orig_jpg_path}")

