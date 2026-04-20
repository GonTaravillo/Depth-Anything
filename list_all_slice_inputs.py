import onnx
import numpy as np
from onnx import numpy_helper

model = onnx.load('depth_anything_nano_epoch10_224x224.onnx')
graph = model.graph

print("All nodes of type Slice:")
for n in graph.node:
    if n.op_type == 'Slice':
        print(f"Node: {n.name}, Inputs: {list(n.input)}")

print("\nAll nodes of type Constant (first 10):")
count = 0
for n in graph.node:
    if n.op_type == 'Constant':
        print(f"Constant Output: {list(n.output)}")
        count += 1
        if count > 10: break
