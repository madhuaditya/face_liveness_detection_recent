import os
import tensorflow as tf
import tf2onnx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KERAS_MODEL_PATH = os.path.join(BASE_DIR, "../models/new_liveness_model.keras")
ONNX_MODEL_PATH  = os.path.join(BASE_DIR, "../models/new_liveness_model.onnx")

print(f"Loading Keras model from: {KERAS_MODEL_PATH}")
model = tf.keras.models.load_model(KERAS_MODEL_PATH)

# ---------- PATCH for newer Keras versions ----------
# tf2onnx expects `model.output_names`, but newer tf.keras doesn't always define it.
if not hasattr(model, "output_names"):
    # Derive simple output names from model.outputs
    try:
        model.output_names = [t.name.split(":")[0] for t in model.outputs]
    except Exception as e:
        print("Warning: could not infer output_names, using default name 'output'. Error:", e)
        model.output_names = ["output"]
# ---------------------------------------------------

# Define input signature: same as training (None, 64, 64, 3)
spec = (tf.TensorSpec((None, 64, 64, 3), tf.float32, name="input"),)

print("Converting Keras model to ONNX...")
model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path=ONNX_MODEL_PATH
)

print(f"✅ ONNX model saved at: {ONNX_MODEL_PATH}")
