import onnx

model = onnx.load('depth_anything_nano_epoch10_112x112.onnx')

for init in model.graph.initializer:
    if init.name == '/Cast_output_0':
        print(f"Initializer: {init.name}, shape: {init.dims}")

for n in model.graph.node:
    if '/Cast_output_0' in n.input:
        print(f"Node '{n.name}' uses '/Cast_output_0'")
