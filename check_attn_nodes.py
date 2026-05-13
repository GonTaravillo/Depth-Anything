import onnx

model = onnx.load('depth_anything_nano_epoch10_224x224.onnx')

print("Nodes in blocks.0.attn:")
for n in model.graph.node:
    if n.name.startswith('/blocks.0/attn/'):
        print(f"{n.name} ({n.op_type}):")
        print(f"  inputs: {n.input}")
        print(f"  outputs: {n.output}")
