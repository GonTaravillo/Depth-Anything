import onnx
import numpy as np
from onnx import numpy_helper

# Cargar el ONNX ANTES de aplicar el fix (regenerar fresco)
import subprocess
subprocess.run(["venv/bin/python3", "tools/export_onnx.py"], capture_output=True)

model = onnx.load('depth_anything_nano_epoch10_224x224.onnx')
init_map = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}

shape_out = '/patch_embed/Shape_output_0'
for n in model.graph.node:
    if any(shape_out in inp for inp in n.input):
        print(f"\nNodo: {n.op_type} '{n.name}'")
        print(f"  inputs: {list(n.input)}")
        print(f"  outputs: {list(n.output)}")
        for inp in n.input:
            if inp in init_map:
                print(f"    init {inp} = {init_map[inp]}")
            elif inp != shape_out:
                # Buscar si es constante via nodo Constant
                for n2 in model.graph.node:
                    if n2.op_type == 'Constant' and inp in n2.output:
                        val = n2.attribute[0].t
                        print(f"    const {inp} = {numpy_helper.to_array(val)}")
