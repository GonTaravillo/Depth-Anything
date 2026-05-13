import onnx
import onnxruntime as ort
import numpy as np

model_path = 'depth_anything_nano_epoch10_112x112.onnx'
sess = ort.InferenceSession(model_path)
input_name = sess.get_inputs()[0].name
dummy = np.random.randn(1, 3, 112, 112).astype(np.float32)

res = sess.run(None, {input_name: dummy})

# To get intermediate shapes, we can enable all outputs
model = onnx.load(model_path)
for node in model.graph.node:
    for out in node.output:
        if out not in [o.name for o in model.graph.output]:
            model.graph.output.extend([onnx.ValueInfoProto(name=out)])
onnx.save(model, 'debug_model.onnx')

sess = ort.InferenceSession('debug_model.onnx')
out_names = [o.name for o in sess.get_outputs()]
outputs = sess.run(out_names, {input_name: dummy})

for name, out in zip(out_names, outputs):
    if hasattr(out, 'shape') and len(out.shape) > 1 and out.shape[1] == 257:
        print(f"Node output '{name}' has shape {out.shape}")
