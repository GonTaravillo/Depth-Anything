import onnx
from onnx import numpy_helper
import numpy as np

model = onnx.load('depth_anything_nano_random_sim.onnx')
for node in model.graph.node:
    if node.op_type == 'Constant':
        val = numpy_helper.to_array(node.attribute[0].t)
        if np.any(val == 14):
            print("Constant with 14:", node.name, val.shape, val)
