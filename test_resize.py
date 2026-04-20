import onnx
model = onnx.load('depth_anything_nano_epoch10_224x224.onnx')
resizes = [n for n in model.graph.node if n.op_type == 'Resize']
print(f"Total resizes: {len(resizes)}")
init_names = [i.name for i in model.graph.initializer]
for r in resizes:
    sizes_is_init = len(r.input) > 3 and r.input[3] in init_names
    scales_is_init = len(r.input) > 2 and r.input[2] in init_names
    print(r.name, r.input, f"sizes_init: {sizes_is_init}, scales_init: {scales_is_init}")
