import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
import tensorflow as tf

# Define paths
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, '..', 'outputs', 'vegetation_model.h5')
data_dir = os.path.join(base_dir, '..', 'data')
output_dir = os.path.join(base_dir, '..', 'outputs')

os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

# Load model
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Error: Model file not found at {model_path}")

model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully.")

# Get all images from `data/`
image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    raise ValueError("❌ No images found in the data folder!")

print(f"🟢 Found {len(image_files)} images. Processing...")

for img_file in image_files:
    img_path = os.path.join(data_dir, img_file)

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Error loading image: {img_path}")
        continue

    # Resize and normalize the image
    img_resized = cv2.resize(img, (256, 256))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)  # Add batch dimension

    # Predict vegetation mask
    mask_pred = model.predict(img_batch)[0, ..., 0]  # Get first channel
    mask_pred = (mask_pred > 0.5).astype(np.uint8) * 255  # Convert to binary mask

    # Save predicted mask
    output_mask_path = os.path.join(output_dir, f"predicted_mask_{img_file}")
    cv2.imwrite(output_mask_path, mask_pred)

    print(f"✅ Processed: {img_file} → {output_mask_path}")

print("🎉 All images processed successfully!")
