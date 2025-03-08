import os
import cv2
import numpy as np

# Define paths
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')

os.makedirs(output_dir, exist_ok=True)

# List all image files in the data directory
image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
processed_images = []
processed_masks = []

for image_file in image_files:
    img_path = os.path.join(data_dir, image_file)
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Error: {img_path} not found or cannot be read!")
        continue  # Skip to next image

    # Resize to 256x256 for consistency
    img_resized = cv2.resize(img, (256, 256))

    # Normalize image to range [0,1]
    img_normalized = img_resized / 255.0
    processed_images.append(img_normalized)

    # Convert to HSV for vegetation segmentation
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    
    # Define vegetation range in HSV (adjust if needed)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)  # Extract vegetation
    mask_resized = cv2.resize(mask, (256, 256))  # Resize mask
    mask_normalized = mask_resized / 255.0  # Normalize to [0,1]
    
    processed_masks.append(mask_normalized)

    # Save processed image and mask
    img_name = os.path.splitext(image_file)[0]  # Extract filename without extension
    cv2.imwrite(os.path.join(output_dir, f"normalized_{img_name}.png"), (img_normalized * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, f"mask_{img_name}.png"), (mask_normalized * 255).astype(np.uint8))

# Convert to NumPy arrays
if processed_images:
    processed_images = np.array(processed_images, dtype=np.float32)
    processed_masks = np.array(processed_masks, dtype=np.float32)
    
    # Save as .npy files in the outputs folder
    np.save(os.path.join(output_dir, 'train_images.npy'), processed_images)
    np.save(os.path.join(output_dir, 'train_masks.npy'), processed_masks)
    
    print(f"✅ Processed {len(processed_images)} images successfully!")
    print("✅ train_images.npy and train_masks.npy saved in outputs folder")
else:
    print("❌ No images were processed!")
