import onnx

model = onnx.load('depth_anything_nano_epoch10_112x112.onnx')

for n in model.graph.node:
    if '/Cast_output_0' in n.output:
        print(f"Node: {n.name}")
        print(f"  inputs: {n.input}")
