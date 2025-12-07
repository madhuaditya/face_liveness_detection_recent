"""
Train MobileNet-based anti-spoof model (transfer learning).
Expect dataset folder structure:
dataset/
    train/
        real/
        fake/
    val/
        real/
        fake/

Saves: models/liveness_mobilenet.keras
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import argparse

# --------- Arguments ----------
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="../dataset", help="dataset root")
parser.add_argument("--out", type=str, default="../models/liveness_mobilenet.keras", help="output model path")
parser.add_argument("--img", type=int, default=224, help="image size (square)")
parser.add_argument("--bs", type=int, default=64, help="batch size")
parser.add_argument("--epochs", type=int, default=20, help="epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="initial learning rate")
parser.add_argument("--finetune_after", type=int, default=50, help="unfreeze after N layers (0=no finetune)")
args = parser.parse_args()

DATA_ROOT = os.path.abspath(args.data)
IMG_SIZE = (args.img, args.img)
BATCH_SIZE = args.bs
EPOCHS = args.epochs
MODEL_PATH = os.path.abspath(args.out)
LR = args.lr

# --------- Setup strategy (single or multi-GPU) ----------
try:
    # Mixed MultiWorker / MultiGPU friendly: detect GPUs
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()
except Exception:
    strategy = tf.distribute.get_strategy()

print("Using strategy:", type(strategy).__name__)

# Enable mixed precision if GPU available
try:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    if tf.config.list_physical_devices("GPU"):
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision enabled")
except Exception:
    pass

# --------- Data pipeline (tf.data via image_dataset_from_directory) ----------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_ROOT, "train"),
    labels="inferred",
    label_mode="binary",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_ROOT, "val"),
    labels="inferred",
    label_mode="binary",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False
)

# Prefetch / caching
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --------- Augmentation (Keras layers) ----------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.08),
    layers.RandomContrast(0.08),
], name="data_augmentation")

# --------- Model builder (MobileNetV3 small fallback) ----------
def build_model(input_shape=(args.img, args.img, 3)):
    try:
        # try MobileNetV3Small if available
        base = tf.keras.applications.MobileNetV3Small(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    except Exception:
        # fallback to MobileNetV2
        base = tf.keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )

    base.trainable = False  # freeze backbone initially

    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # works for both
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    # If mixed_precision policy is set, final dense should be float32 for stable loss scaling
    head = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model = models.Model(inputs, head)
    return model, base

with strategy.scope():
    model, base_backbone = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    opt = optimizers.Adam(learning_rate=LR)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

model.summary()

# --------- Callbacks ----------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
cbks = [
    callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True, save_weights_only=False),
    callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=6, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=3, min_lr=1e-7, mode="max"),
    callbacks.TensorBoard(log_dir=os.path.join("logs", "mobilenet_antispoof"))
]

# --------- Train (stage 1: feature-extractor) ----------
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbks)

# --------- Optional fine-tuning ----------
# unfreeze last N layers if requested
if args.finetune_after and args.finetune_after > 0:
    print("Unfreezing backbone for fine-tuning...")
    base_backbone.trainable = True
    # Optionally freeze first layers, unfreeze last N
    if args.finetune_after > 0:
        for layer in base_backbone.layers[:-args.finetune_after]:
            layer.trainable = False
    # recompile with lower LR
    with strategy.scope():
        model.compile(optimizer=optimizers.Adam(learning_rate=LR * 0.1),
                      loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    # fine tune
    ft_epochs = int(EPOCHS * 0.5) if EPOCHS>2 else 3
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS + ft_epochs, callbacks=cbks)

# final save (keras format)
model.save(MODEL_PATH)
print("Model saved to:", MODEL_PATH)
