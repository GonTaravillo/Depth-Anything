import onnx
import numpy as np
from onnx import numpy_helper

model = onnx.load('depth_anything_nano_epoch10_224x224.onnx')

# Encontrar el Slice que consume Shape_output_0
shape_out = '/patch_embed/Shape_output_0'
for n in model.graph.node:
    if n.op_type == 'Slice' and shape_out in n.input:
        print(f"Slice: {n.name}")
        print(f"  inputs: {list(n.input)}")
        print(f"  outputs: {list(n.output)}")
        # Ver los initializers de starts/ends/axes
        init_map = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        for inp in n.input[1:]:
            if inp in init_map:
                print(f"  {inp} = {init_map[inp]}")
        # Quién consume la salida del Slice?
        slice_out = list(n.output)[0]
        for n2 in model.graph.node:
            if slice_out in n2.input:
                print(f"  -> consumido por: {n2.op_type} '{n2.name}' inputs={list(n2.input)}")
