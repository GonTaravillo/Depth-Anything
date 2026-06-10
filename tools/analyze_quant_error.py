import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.abspath('esp-ppq'))
from esp_ppq import *
from esp_ppq.api import *

def get_transform(target_size=112):
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def collate_fn(batch):
    return torch.stack(batch).to('cuda' if torch.cuda.is_available() else 'cpu')

# Load calibration image
image = Image.open('rpi_deploy/orig_0.jpg').convert("RGB")
tensor = get_transform()(image)

# Load the unquantized ONNX model into PPQ
onnx_file = 'depth_anything_nano_112x112_trained_clean.onnx'
target = TargetPlatform.ESP32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading ONNX into PPQ...")
graph = load_onnx_graph(onnx_import_file=onnx_file)
quantizer = PpqQuantizer(graph=graph, collate_fn=collate_fn, target=target)
dispatching = quantizer.dispatching_table

# Load the quantized model
print("Quantizing in memory for error analysis...")
dataloader = [tensor] * 32
with ENABLE_CUDA_KERNEL():
    quantized_graph = quantize_onnx_model(
        onnx_import_file=onnx_file,
        calib_dataloader=dataloader,
        calib_steps=1,
        input_shape=[1, 3, 112, 112],
        target_platform=target,
        device=device,
        collate_fn=collate_fn
    )

    print("Running Error Analysis (Signal-to-Noise Ratio)...")
    # Execute the graph
    executor = TorchExecutor(graph=quantized_graph, device=device)
    inputs = [tensor.unsqueeze(0).to(device)]
    
    # We want to see the error at the output
    from ppq.quantization.analyze import graphwise_error_analyse
    reports = graphwise_error_analyse(
        graph=quantized_graph, 
        running_device=device, 
        dataloader=dataloader, 
        collate_fn=collate_fn,
        steps=1
    )
    
