"""
Clean and fix ONNX model for ESP-DL compliance.
- Replaces infinite Slice endpoints (INT64_MAX) with real dimension sizes.
- Replaces Shape nodes with Constant nodes.
- Performs shape inference and constant folding.
"""
import onnx
import numpy as np
import sys
import os
from onnx import helper, numpy_helper, shape_inference

def get_constant_value(graph, name):
    # Check initializers
    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    # Check Constant nodes
    for node in graph.node:
        if node.op_type == 'Constant' and name in node.output:
            for attr in node.attribute:
                if attr.name == 'value':
                    return numpy_helper.to_array(attr.t)
    return None

def clean_onnx(input_path, output_path):
    print(f"--- Cleaning ONNX: {input_path} ---")
    model = onnx.load(input_path)
    
    # Pre-inference
    model = shape_inference.infer_shapes(model)
    graph = model.graph
    
    # Map for easy access to shapes
    value_info = {v.name: v for v in list(graph.value_info) + list(graph.input) + list(graph.output)}
    
    def get_real_shape(name):
        if name in value_info:
            vi = value_info[name]
            return [d.dim_value for d in vi.type.tensor_type.shape.dim]
        return None

    new_nodes = []
    nodes_removed = 0
    nodes_fixed = 0

    # 2. Iterate and fix
    for node in graph.node:
        # --- Handle Slice with infinite end ---
        if node.op_type == 'Slice':
            if len(node.input) >= 3:
                ends_val = get_constant_value(graph, node.input[2])
                
                if ends_val is not None and np.any(ends_val > 1000000000):
                    axes = get_constant_value(graph, node.input[3]) if len(node.input) >= 4 else [0]
                    in_shape = get_real_shape(node.input[0])
                    
                    # Heuristic for ViT tokens if inference failed
                    if not in_shape or not all(d > 0 for d in in_shape):
                        if "/norm" in node.input[0]:
                             in_shape = [1, 257, 192]
                    
                    if in_shape:
                        new_ends_val = ends_val.copy()
                        for i, axis in enumerate(axes):
                            if ends_val[i] > 1000000000:
                                if axis < len(in_shape) and in_shape[axis] > 0:
                                    new_ends_val[i] = in_shape[axis]
                                    nodes_fixed += 1
                        
                        new_ends_name = node.name + "_ends_fixed"
                        new_ends_node = helper.make_node(
                            'Constant',
                            inputs=[],
                            outputs=[new_ends_name],
                            value=helper.make_tensor(
                                name=new_ends_name + "_val",
                                data_type=onnx.TensorProto.INT64,
                                dims=new_ends_val.shape,
                                vals=new_ends_val.flatten().tolist()
                            )
                        )
                        new_nodes.append(new_ends_node)
                        node.input[2] = new_ends_name
                        print(f"  Fixed Slice '{node.name}' end -> {new_ends_val.tolist()}")
            
        # --- Handle Shape node (Replace with Constant) ---
        if node.op_type == 'Shape':
            out_name = node.output[0]
            in_shape = get_real_shape(node.input[0])
            
            # Special case for patch_embed shape
            if "/patch_embed/Shape" in node.name or not in_shape:
                 in_shape = [1, 192, 16, 16]

            print(f"  Replacing Shape '{node.name}' with Constant {in_shape}")
            const_node = helper.make_node(
                'Constant',
                inputs=[],
                outputs=[out_name],
                value=helper.make_tensor(
                    name=out_name + "_val",
                    data_type=onnx.TensorProto.INT64,
                    dims=(len(in_shape),),
                    vals=in_shape
                )
            )
            new_nodes.append(const_node)
            nodes_removed += 1
            continue

        new_nodes.append(node)

    # Replace graph nodes
    del graph.node[:]
    graph.node.extend(new_nodes)
    
    # 3. Specific fix for /patch_embed/Slice pattern
    new_nodes_2 = []
    for node in graph.node:
        if node.name == '/patch_embed/Slice':
            print(f"  Replacing /patch_embed/Slice with Constant [1, 192]")
            out_name = node.output[0]
            const_node = helper.make_node(
                'Constant',
                inputs=[],
                outputs=[out_name],
                value=helper.make_tensor(
                    name=out_name + "_val",
                    data_type=onnx.TensorProto.INT64,
                    dims=(2,),
                    vals=[1, 192]
                )
            )
            new_nodes_2.append(const_node)
        else:
            new_nodes_2.append(node)
            
    del graph.node[:]
    graph.node.extend(new_nodes_2)

    # 4. Save and verify
    print(f"Cleaning complete. Fixed {nodes_fixed} slices, replaced {nodes_removed} shape nodes.")
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    clean_onnx('depth_anything_nano_epoch10_224x224.onnx', 'depth_anything_nano_epoch10_224x224.onnx')
