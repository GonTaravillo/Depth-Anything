import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'depth_anything_nano_112x112_trained_clean.onnx'
model_quant = 'depth_anything_nano_112x112_int8.onnx'

quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
print(f"Cuantización dinámica completada: {model_quant}")
