import tensorflow as tf
import tf2onnx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to input .keras model")
parser.add_argument("--output", required=True, help="Path to output .onnx model")
args = parser.parse_args()

print("[INFO] Loading Keras model:", args.input)
model = tf.keras.models.load_model(args.input)

spec = (tf.TensorSpec(model.inputs[0].shape, tf.float32, name="input"),)

print("[INFO] Converting to ONNX...")
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

print("[INFO] Saving:", args.output)
with open(args.output, "wb") as f:
    f.write(onnx_model.SerializeToString())

print("[DONE] Conversion successful!")
