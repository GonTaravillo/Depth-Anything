import cv2
import numpy as np
img = cv2.imread("PC_Inference/result_image_0.png")
print("Image shape:", img.shape)
print("Image left part mean:", np.mean(img[:, :448]))
print("Image right part mean:", np.mean(img[:, 488:]))
