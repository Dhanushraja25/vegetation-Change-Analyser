import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Input
import numpy as np


print(tf.__version__)

# Define U-Net Model
def build_unet(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)

    # Decoder
    u1 = UpSampling2D((2, 2))(c3)
    m1 = Concatenate()([u1, c2])
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(m1)

    u2 = UpSampling2D((2, 2))(c4)
    m2 = Concatenate()([u2, c1])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(m2)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = keras.models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Load and preprocess data
data_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')  # Updated path to outputs
train_images = np.load(os.path.join(data_dir, 'train_images.npy'))
train_masks = np.load(os.path.join(data_dir, 'train_masks.npy'))

# Reshape the masks to ensure they have a single channel
train_masks = train_masks.reshape(-1, 256, 256, 1)

# Train model
model = build_unet()
model.fit(train_images, train_masks, epochs=10, batch_size=8)

# Save model
model.save(os.path.join(data_dir, "vegetation_model.h5"))
print("Model trained and saved.")
