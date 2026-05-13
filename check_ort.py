import onnx
import onnxruntime as ort
import numpy as np

model_path = 'depth_anything_nano_epoch10_112x112.onnx'
sess = ort.InferenceSession(model_path)
input_name = sess.get_inputs()[0].name
dummy = np.random.randn(1, 3, 112, 112).astype(np.float32)

try:
    sess.run(None, {input_name: dummy})
    print("Inference successful!")
except Exception as e:
    print("Error during inference:", e)
