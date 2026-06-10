import cv2
import numpy as np
import os
import onnxruntime as ort
from PIL import Image
from transformers import pipeline

def normalize_image(image_rgb):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    img = image_rgb.astype(np.float32) / 255.0
    img = (img - mean) / std
    img = img.transpose(2, 0, 1) # HWC to CHW
    img = np.expand_dims(img, axis=0) # CHW to BCHW
    return img

def main():
    onnx_path = 'rpi_deploy/depth_anything_nano_112x112_trained_clean.onnx'
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model {onnx_path} not found.")
        return
        
    print("Loading ONNX Nano model...")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, session_options, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    print("Loading original Depth-Anything ViT-B model from HuggingFace...")
    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf", device=-1)

    for i in range(5):
        orig_path = f'rpi_deploy/orig_{i}.jpg'
        esp32_path = f'/media/gonzalo/games/depth_out_{i}.jpg'
        
        if not os.path.exists(orig_path):
            print(f"Skipping {orig_path}, not found.")
            continue
            
        if not os.path.exists(esp32_path):
            print(f"Skipping {esp32_path}, not found. Ensure SD card is mounted.")
            continue

        print(f"Processing image {i}...")

        # 1. Original Image (resized to 112x112)
        image_bgr = cv2.imread(orig_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (112, 112))
        
        # 2. Original ViT-B Inference
        pil_image = Image.fromarray(image_rgb)
        depth_vitb_pil = pipe(pil_image)["depth"]
        depth_vitb_np = np.array(depth_vitb_pil)
        
        # Resize ViT-B depth map to 112x112 and colorize
        depth_vitb_resized = cv2.resize(depth_vitb_np, (112, 112))
        d_min, d_max = depth_vitb_resized.min(), depth_vitb_resized.max()
        depth_vitb_norm = (depth_vitb_resized - d_min) / (d_max - d_min + 1e-8) * 255.0
        depth_vitb_norm = depth_vitb_norm.astype(np.uint8)
        depth_colored_vitb = cv2.applyColorMap(depth_vitb_norm, cv2.COLORMAP_INFERNO)

        # 3. FP32 PC Inference (Nano)
        input_tensor = normalize_image(image_resized)
        outputs = session.run(None, {input_name: input_tensor})
        depth = np.squeeze(outputs[0])
        
        d_min, d_max = depth.min(), depth.max()
        depth_norm = (depth - d_min) / (d_max - d_min + 1e-8) * 255.0
        depth_norm = depth_norm.astype(np.uint8)
        depth_colored_fp32 = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        
        # 4. INT8 ESP32 Result (Nano)
        depth_colored_esp32 = cv2.imread(esp32_path)
        if depth_colored_esp32.shape[:2] != (112, 112):
            depth_colored_esp32 = cv2.resize(depth_colored_esp32, (112, 112))
            
        # Add labels
        orig_bgr = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
        
        # We need more space for labels, let's scale everything up by 3x for visibility
        scale = 3
        orig_bgr = cv2.resize(orig_bgr, (112*scale, 112*scale), interpolation=cv2.INTER_NEAREST)
        depth_colored_vitb = cv2.resize(depth_colored_vitb, (112*scale, 112*scale), interpolation=cv2.INTER_NEAREST)
        depth_colored_fp32 = cv2.resize(depth_colored_fp32, (112*scale, 112*scale), interpolation=cv2.INTER_NEAREST)
        depth_colored_esp32 = cv2.resize(depth_colored_esp32, (112*scale, 112*scale), interpolation=cv2.INTER_NEAREST)
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(orig_bgr, "Original", (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(depth_colored_vitb, "ViT-Base PC", (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(depth_colored_fp32, "Nano PC FP32", (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(depth_colored_esp32, "Nano ESP32 INT8", (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Concatenate horizontally
        separator = np.zeros((112*scale, 10, 3), dtype=np.uint8)
        combined = cv2.hconcat([orig_bgr, separator, depth_colored_vitb, separator, depth_colored_fp32, separator, depth_colored_esp32])
        
        out_path = f'comparison_full_{i}.png'
        cv2.imwrite(out_path, combined)
        print(f"Saved comparison to {out_path}")

if __name__ == "__main__":
    main()
