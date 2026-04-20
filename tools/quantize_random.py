import os
import sys
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
import glob

# Add esp-ppq to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'esp-ppq')))

from esp_ppq import *
from esp_ppq.api import *
from esp_ppq.api import espdl_quantize_onnx

def load_calibration_dataset(data_dir, num_samples=32, target_size=224):
    image_paths = glob.glob(os.path.join(data_dir, "*.jpg"))
    if len(image_paths) == 0:
        print(f"No JPG files found in {data_dir}. Check paths.")
        return []
        
    image_paths = image_paths[:num_samples]
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = []
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert("RGB")
            image = transform(image)
            dataset.append(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            
    print(f"Loaded {len(dataset)} calibration samples from {data_dir}.")
    return dataset

def main():
    WORKING_DIRECTORY = 'working_quant'
    os.makedirs(WORKING_DIRECTORY, exist_ok=True)
    
    TARGET = "esp32s3"
    NUM_OF_BITS = 8
    NETWORK_INPUTSHAPE = [1, 3, 192, 192]
    EXECUTING_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    onnx_file = 'depth_anything_nano_random_sim_clean.onnx'
    espdl_file = os.path.join(WORKING_DIRECTORY, 'depth_anything_pico.espdl')
    
    if not os.path.exists(onnx_file):
        raise FileNotFoundError(f"ONNX model file {onnx_file} not found.")
        
    base_data = os.path.join('data', 'open_images_v7')
    data_dirs = [
        os.path.join(base_data, 'test', 'data'),
        os.path.join(base_data, 'validation', 'data'),
        os.path.join(base_data, 'train', 'data')
    ]
    
    data_dir = None
    for d in data_dirs:
        if os.path.exists(d) and len(glob.glob(os.path.join(d, "*.jpg"))) > 0:
            data_dir = d
            break
            
    if data_dir is None:
         # Create a dummy calibration dataset if no real data is found since we just want to test if it fits/runs
         print("No calibration images found! Using dummy random data for testing.")
         dataset = [torch.randn(3, 192, 192) for _ in range(8)]
    else:
         dataset = load_calibration_dataset(data_dir, num_samples=32, target_size=192)
         
    if len(dataset) == 0:
        dataset = [torch.randn(3, 192, 192) for _ in range(8)]
        
    def collate_fn(batch):
        if isinstance(batch, list):
             return torch.stack(batch).to(EXECUTING_DEVICE)
        if isinstance(batch, torch.Tensor):
             return batch.to(EXECUTING_DEVICE)
        return batch

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        
    print(f"🚀 Iniciando cuantización para {TARGET}...")

    quant_ppq_graph = espdl_quantize_onnx(
        onnx_import_file=onnx_file,
        espdl_export_file=espdl_file,
        calib_dataloader=dataloader,
        calib_steps=len(dataset),
        input_shape=NETWORK_INPUTSHAPE,
        inputs=None,
        target=TARGET,
        num_of_bits=NUM_OF_BITS,
        collate_fn=collate_fn,
        dispatching_override=None,
        device=EXECUTING_DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,
    )
    
    print("\n🎉 ¡PROCESO TERMINADO!")
    print(f"Generado: {espdl_file}")

if __name__ == "__main__":
    main()
