import onnx; model = onnx.load('depth_anything_nano_random_sim.onnx'); [print(n.name, n.op_type, [i for i in n.input]) for n in model.graph.node if n.op_type == 'Slice']
