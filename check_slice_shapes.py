import onnx
import numpy as np
from onnx import numpy_helper

# Usar el modelo que se guarda al final de quantize_espdl.py si es posible,
# pero por ahora el onnx local.
model = onnx.load('depth_anything_nano_epoch10_224x224.onnx')
import onnx.shape_inference
inferred = onnx.shape_inference.infer_shapes(model)
value_info = {v.name: v for v in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output)}

print("Detailed Slice input shapes:")
for n in inferred.graph.node:
    if n.op_type == 'Slice':
        print(f"\nNode: {n.name}")
        in_name = n.input[0]
        if in_name in value_info:
            vi = value_info[in_name]
            shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            print(f"  Input '{in_name}' shape: {shape}")
        else:
            print(f"  Input '{in_name}' shape: NOT FOUND")

