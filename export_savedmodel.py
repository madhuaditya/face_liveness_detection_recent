import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

model = tf.keras.models.load_model(args.input)
tf.saved_model.save(model, args.output)

print("SavedModel exported to:", args.output)
