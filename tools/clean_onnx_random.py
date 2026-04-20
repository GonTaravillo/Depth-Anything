import onnx
import numpy as np
import sys
from onnx import helper, numpy_helper, shape_inference

def get_constant_value(graph, name):
    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    for node in graph.node:
        if node.op_type == 'Constant' and name in node.output:
            for attr in node.attribute:
                if attr.name == 'value':
                    return numpy_helper.to_array(attr.t)
    return None

def clean_onnx(input_path, output_path):
    print(f"--- Cleaning ONNX Slices for ESP-DL: {input_path} ---")
    model = onnx.load(input_path)
    model = shape_inference.infer_shapes(model)
    graph = model.graph
    
    value_info = {v.name: v for v in list(graph.value_info) + list(graph.input) + list(graph.output)}
    
    def get_real_shape(name):
        if name in value_info:
            vi = value_info[name]
            return [d.dim_value for d in vi.type.tensor_type.shape.dim]
        return None

    new_nodes = []
    nodes_fixed = 0

    for node in graph.node:
        if node.op_type == 'Slice':
            if len(node.input) >= 3:
                ends_val = get_constant_value(graph, node.input[2])
                if ends_val is not None and np.any(ends_val > 10000000):
                    axes = get_constant_value(graph, node.input[3]) if len(node.input) >= 4 else [0]
                    in_shape = get_real_shape(node.input[0])
                    print(f"Found infinite Slice op on {node.input[0]} with shape {in_shape}")
                    
                    if not in_shape or not all(d > 0 for d in in_shape):
                        in_shape = [1, 145, 192] # Hardcoded fallback for random nano POS token length at 192x192 (12x12 grid = 144 + 1)
                        
                    if in_shape:
                        new_ends_val = ends_val.copy()
                        for i, axis in enumerate(axes):
                            if ends_val[i] > 10000000:
                                if axis < len(in_shape) and in_shape[axis] > 0:
                                    new_ends_val[i] = in_shape[axis]
                                    nodes_fixed += 1
                        
                        new_ends_name = node.name + "_ends_fixed"
                        new_ends_node = helper.make_node(
                            'Constant', inputs=[], outputs=[new_ends_name],
                            value=helper.make_tensor(name=new_ends_name + "_val", data_type=onnx.TensorProto.INT64, dims=new_ends_val.shape, vals=new_ends_val.flatten().tolist())
                        )
                        new_nodes.append(new_ends_node)
                        node.input[2] = new_ends_name
                        print(f"  Fixed Slice '{node.name}' end -> {new_ends_val.tolist()}")
        
        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    
    print(f"Cleaning complete. Fixed {nodes_fixed} slices.")
    onnx.checker.check_model(model)
    onnx.save(model, output_path)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        clean_onnx(sys.argv[1], sys.argv[2])
    else:
        clean_onnx('depth_anything_nano_random_sim.onnx', 'depth_anything_nano_random_sim_clean.onnx')
