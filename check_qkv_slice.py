import onnx
import onnx.shape_inference
from onnx import numpy_helper

model = onnx.load('depth_anything_nano_epoch10_224x224.onnx')
model = onnx.shape_inference.infer_shapes(model)

def get_constant_value(graph, name):
    for init in graph.initializer:
        if init.name == name: return numpy_helper.to_array(init)
    for node in graph.node:
        if node.op_type == 'Constant' and name in node.output:
            for attr in node.attribute:
                if attr.name == 'value': return numpy_helper.to_array(attr.t)
    return None

for n in model.graph.node:
    if n.name == '/blocks.0/attn/Slice':
        print(f"Node: {n.name}")
        for i, inp in enumerate(n.input):
            val = get_constant_value(model.graph, inp)
            print(f"  Input[{i}] '{inp}': constant={val}")
