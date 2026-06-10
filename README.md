# Depth-Anything Nano (Edge AI Deployment)

This repository contains the complete end-to-end pipeline for training, exporting, quantizing, and deploying a highly compressed "Nano" version (~500K parameters) of the Depth-Anything ViT architecture onto an ESP32-S3 microcontroller.

This project is part of a Final Degree Project (PFG) focusing on extreme Edge AI compression for depth estimation in resource-constrained environments.

## Repository Structure

- `model/`: PyTorch definitions of the Nano architecture (`dpt.py`, `blocks.py`).
- `training/`: Scripts used to train the model from scratch on NYU Depth V2 / OpenImages.
- `tools/`: Utility scripts to test, evaluate, export to ONNX, and quantize to ESP-DL INT8.
- `firmware/`: ESP-IDF C++ project ready to be flashed to an ESP32-S3 (Korvo-2).
- `pretrained_models/`: The final trained ONNX model and the quantized `model.espdl` file.

## Getting Started

### 1. Training (Optional)
If you wish to retrain the model from scratch, ensure you have the required datasets and run:
```bash
python training/train_112.py
```

### 2. Export & Quantization
To convert the PyTorch `.pth` weights to ONNX and then to ESP-DL format:
```bash
python tools/export_onnx.py
python tools/quantize_espdl.py
```
*Note: Quantization requires the `esp-ppq` framework.*

### 3. Evaluation
To evaluate the model against Ground Truth (RMSE, AbsRel):
```bash
python tools/evaluate_with_gt.py     # NYU Depth V2
python tools/evaluate_with_diode.py  # DIODE Dataset
```

### 4. ESP32-S3 Deployment
The firmware uses the ESP-DL library to run the `model.espdl` in static INT8 precision.
1. Copy test images to `firmware/main/` and run `python tools/img_to_bin.py` to convert them to RGB565/NHWC binary arrays.
2. Build and flash the firmware using ESP-IDF:
```bash
cd firmware
idf.py set-target esp32s3
idf.py build
idf.py flash monitor
```
The serial monitor will output the predicted depth ranges and performance profiling metrics.

## Performance
- **Inference Time (ESP32-S3):** ~2970 ms per frame (112x112 resolution).
- **RAM Usage:** Fully fits within the internal SRAM + PSRAM using static allocation.
- **Accuracy:** AbsRel ~15% on DIODE Indoor dataset (compared to 5% of the original 100M parameter model).
