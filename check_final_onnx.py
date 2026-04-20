import onnx
import numpy as np
from onnx import numpy_helper

model = onnx.load('depth_anything_nano_epoch10_224x224.onnx')
graph = model.graph

def get_val(name):
    for i in graph.initializer:
        if i.name == name: return numpy_helper.to_array(i)
    for n in graph.node:
        if n.op_type == 'Constant' and name in n.output:
            for a in n.attribute:
                if a.name == 'value': return numpy_helper.to_array(a.t)
    return None

print("Checking all Slice nodes:")
for n in graph.node:
    if n.op_type == 'Slice':
        print(f"\nNode: {n.name}")
        for i, inp in enumerate(n.input):
            val = get_val(inp)
            if val is not None:
                print(f"  input[{i}] ({inp}) = {val}")
