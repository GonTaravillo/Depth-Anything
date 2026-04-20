import onnx
from onnx import numpy_helper

model = onnx.load('depth_anything_nano_random_sim_clean.onnx')
for node in model.graph.node:
    if node.op_type == 'Add':
        shape0 = "Unknown"
        shape1 = "Unknown"
        # look up value info
        for vi in list(model.graph.value_info) + list(model.graph.output) + list(model.graph.input):
            if vi.name == node.input[0]:
                shape0 = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            if vi.name == node.input[1]:
                shape1 = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        # check initializers
        for init in model.graph.initializer:
            if init.name == node.input[0]:
                shape0 = init.dims
            if init.name == node.input[1]:
                shape1 = init.dims
                
        print(f"Add node: {node.name}, inputs: {node.input}, shapes: {shape0} + {shape1}")

