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
            # The tensor returned by transform is shape [3, target_size, target_size]
            image = transform(image) # [3, target_size, target_size]
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
    NETWORK_INPUTSHAPE = [1, 3, 112, 112]
    EXECUTING_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    onnx_file = 'depth_anything_nano_112x112_trained_clean.onnx'
    espdl_file = os.path.join(WORKING_DIRECTORY, 'depth_anything_nano_112x112_trained.espdl')
    
    if not os.path.exists(onnx_file):
        raise FileNotFoundError(f"ONNX model file {onnx_file} not found. Please export it first.")
        
    print("Finding calibration dataset directory...")
    # First try test, then val, then train
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
         raise FileNotFoundError("Could not find any directory with .jpg files for calibration.")
             
    dataset = load_calibration_dataset(data_dir, num_samples=32, target_size=112)
    if len(dataset) == 0:
        raise ValueError("Calibration dataset is empty!")
        
    # Collate function needed for this API
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
    print("Este archivo .espdl es el que debes convertir a .h o cargar en el ESP32.")

if __name__ == "__main__":
    main()
