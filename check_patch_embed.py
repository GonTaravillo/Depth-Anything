import onnx
model = onnx.load('depth_anything_nano_epoch10_224x224.onnx')
# Encontrar la salida del Conv del patch_embed
for n in model.graph.node:
    if 'patch_embed' in n.name and n.op_type == 'Conv':
        print(f"Conv: {n.name}, output: {list(n.output)}")
# Ver info de shape de esa salida en value_info
all_shapes = {v.name: v for v in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)}
target = '/patch_embed/proj/Conv_output_0'
if target in all_shapes:
    vi = all_shapes[target]
    dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    print(f"Shape de {target}: {dims}")
