import numpy as np
import cv2
import os

import glob
img_path = glob.glob('data/open_images_v7/train/data/*.jpg')[0]
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (112, 112))
image = image.astype(np.float32) / 255.0

# Normalize like in training
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
image = (image - mean) / std

# HWC to CHW
image = np.transpose(image, (2, 0, 1))

# Write binary
bin_path = 'depth_anything_firmware/main/image.bin'
with open(bin_path, 'wb') as f:
    f.write(image.tobytes())

print(f"Written {image.nbytes} bytes to {bin_path}")
