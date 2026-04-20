import onnx
import numpy as np
from onnx import numpy_helper

model = onnx.load('depth_anything_nano_epoch10_224x224.onnx')
graph = model.graph

# Necesitamos inferir shapes si no están en value_info
import onnx.shape_inference
inferred_model = onnx.shape_inference.infer_shapes(model)
value_info = {v.name: v for v in inferred_model.graph.value_info}
value_info.update({i.name: i for i in inferred_model.graph.input})
value_info.update({o.name: o for o in inferred_model.graph.output})

init_map = {i.name: numpy_helper.to_array(i) for i in graph.initializer}

def get_shape(name):
    if name in value_info:
        return [d.dim_value for d in value_info[name].type.tensor_type.shape.dim]
    return None

print(f"Información detallada de Slice:")
for n in graph.node:
    if n.op_type == 'Slice':
        print(f"\nNodo: {n.name}")
        in_shape = get_shape(n.input[0])
        print(f"  Shape entrada ({n.input[0]}): {in_shape}")
        
        # Check constants
        args = {}
        for i, name in enumerate(n.input):
            if name in init_map:
                args[i] = init_map[name]
        
        # Starts, ends, axes, steps
        starts = args.get(1, [0])[0]
        ends = args.get(2, [0])[0]
        axes = args.get(3, [0])[0]
        steps = args.get(4, [1])[0]
        
        print(f"  Params: starts={starts}, ends={ends}, axes={axes}, steps={steps}")
        
        if ends > 1000000:
             print(f"  [ALERTA] End gigante detectado: {ends}")
             if in_shape and axes < len(in_shape):
                 real_end = in_shape[axes]
                 print(f"  [FIX] Podría ser reemplazado por: {real_end}")

