import onnx
model = onnx.load('depth_anything_nano_random_sim.onnx')
for vi in model.graph.value_info:
    shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    if 14 in shape:
        print("ValueInfo with 14:", vi.name, shape)
for vi in model.graph.output:
    shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    if 14 in shape:
        print("Output with 14:", vi.name, shape)
for node in model.graph.node:
    if node.op_type == 'Slice':
        print(node.name, node.input)
