import onnx
import numpy as np
from onnx import numpy_helper

model = onnx.load('depth_anything_nano_epoch10_224x224.onnx')
graph = model.graph
init_map = {i.name: numpy_helper.to_array(i) for i in graph.initializer}

print(f"Buscando todos los nodos Slice...")
for n in graph.node:
    if n.op_type == 'Slice':
        print(f"\nNodo: {n.name}")
        print(f"  Inputs: {list(n.input)}")
        for i, inp in enumerate(n.input):
            if inp in init_map:
                print(f"    input[{i}] ({inp}) = {init_map[inp]}")
            else:
                # Check if it's an output of a Constant node
                for c in graph.node:
                    if c.op_type == 'Constant' and inp in c.output:
                        for attr in c.attribute:
                            if attr.name == 'value':
                                print(f"    input[{i}] ({inp}) = Constant {numpy_helper.to_array(attr.t)}")

# También buscar nodos Shape por si queda alguno
shape_nodes = [n for n in graph.node if n.op_type == 'Shape']
print(f"\nNodos Shape restantes: {len(shape_nodes)}")
