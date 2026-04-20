import onnx
model = onnx.load('depth_anything_nano_epoch10_224x224.onnx')
shape_nodes = [n for n in model.graph.node if n.op_type == 'Shape']
print(f"Nodos Shape encontrados: {len(shape_nodes)}")
for n in shape_nodes:
    print(f"  {n.name}: input={list(n.input)} output={list(n.output)}")
    # Quién consume la salida de este Shape?
    for consumer in model.graph.node:
        if any(o in consumer.input for o in n.output):
            print(f"    -> consumido por: {consumer.op_type} '{consumer.name}'")
