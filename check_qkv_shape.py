import onnx
import onnx.shape_inference

model = onnx.load('depth_anything_nano_epoch10_224x224.onnx')
model = onnx.shape_inference.infer_shapes(model)
value_info = {v.name: v for v in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)}

for n in model.graph.node:
    if n.name == '/blocks.0/attn/qkv/Add':
        print(f"Node: {n.name}")
        out_name = n.output[0]
        if out_name in value_info:
            vi = value_info[out_name]
            shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            print(f"  Output '{out_name}' shape: {shape}")
        else:
            print(f"  Output '{out_name}' shape: NOT FOUND")
