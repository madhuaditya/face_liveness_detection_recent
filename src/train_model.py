import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "../dataset")
MODEL_PATH = os.path.join(BASE_DIR, "../models/new_liveness_model.keras")

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)



test_gen = test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "test"),   # or "val" if you don't have test
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # IMPORTANT for correct metrics
)

val_gen = val_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "val"),
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
EPOCHS = 10
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# Save model
os.makedirs(os.path.join(BASE_DIR, "../models"), exist_ok=True)
model.save(MODEL_PATH)

print(f"✅ Model saved at {MODEL_PATH}")
