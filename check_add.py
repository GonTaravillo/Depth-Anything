import onnx
import onnx.shape_inference
from onnx import numpy_helper

model = onnx.load('depth_anything_nano_epoch10_112x112.onnx')
model = onnx.shape_inference.infer_shapes(model)
value_info = {v.name: v for v in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)}

for n in model.graph.node:
    if n.name == '/Add':
        print(f"Node: {n.name}")
        for i, inp in enumerate(n.input):
            if inp in value_info:
                vi = value_info[inp]
                shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
                print(f"  Input[{i}] '{inp}' shape: {shape}")
            else:
                print(f"  Input[{i}] '{inp}' shape: NOT FOUND")
