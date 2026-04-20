"""
Reemplaza el patrón Shape -> Slice del patch_embed con una constante fija.
Para input 224x224 con embed_dim=192 y patch_size=14:
  - Shape(/patch_embed/proj/Conv_output_0) = [1, 192, 16, 16]
  - Slice([1,192,16,16], starts=[0], ends=[2], axes=[0]) = [1, 192]
Sustituimos ambos nodos por un solo nodo Constant que produce [1, 192].
"""
import onnx
import numpy as np
from onnx import numpy_helper, helper, TensorProto

model = onnx.load('depth_anything_nano_epoch10_224x224.onnx')
graph = model.graph

# La salida final que necesita el resto del grafo es la del Slice
slice_out_name = '/patch_embed/Slice_output_0'

# Valor que produce Slice(Shape(conv_out), 0, 2): los 2 primeros dims = [batch, embed]
const_value = np.array([1, 192], dtype=np.int64)

# Nodo Constant que reemplaza a Shape + Slice
constant_node = helper.make_node(
    'Constant',
    inputs=[],
    outputs=[slice_out_name],
    value=numpy_helper.from_array(const_value)
)

# Eliminar Shape y Slice, añadir el Constant
shape_node_name = '/patch_embed/Shape'
slice_node_name = '/patch_embed/Slice'

new_nodes = []
for node in graph.node:
    if node.name in (shape_node_name, slice_node_name):
        print(f"Eliminando nodo: {node.op_type} '{node.name}'")
        if node.name == slice_node_name:
            new_nodes.append(constant_node)  # insertar el reemplazo una sola vez
    else:
        new_nodes.append(node)

del graph.node[:]
graph.node.extend(new_nodes)

onnx.checker.check_model(model)
onnx.save(model, 'depth_anything_nano_epoch10_224x224.onnx')
print("Modelo guardado sin Shape+Slice!")
