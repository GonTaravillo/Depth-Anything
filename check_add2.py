import onnx

model = onnx.load('depth_anything_nano_epoch10_112x112.onnx')

for n in model.graph.node:
    if n.name == '/Add':
        print(f"Node: {n.name}")
        print(f"  inputs: {n.input}")
    if n.name == '/Cast':
        print(f"Node: {n.name}")
        print(f"  inputs: {n.input}")
