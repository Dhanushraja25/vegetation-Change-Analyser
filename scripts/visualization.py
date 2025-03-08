import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define paths
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')

# Load images and masks
years = ["2017", "2020", "2024"]
predicted_masks = []

for year in years:
    # Load the original image
    img_path = os.path.join(data_dir, f"{year}.png")
    img = cv2.imread(img_path)

    # Load the predicted mask
    mask_path = os.path.join(output_dir, f"mask_{year}.png")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"❌ Error: Image or mask for {year} not found!")
        continue

    # Append predicted masks for later use (optional)
    predicted_masks.append(mask)

    # Plot original image and mask
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Original Image {year}")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title(f"Predicted Mask {year}")
    plt.axis('off')

    # If you have ground truth masks, load and plot them
    # Uncomment and modify the following lines if you have ground truth masks
    # gt_mask_path = os.path.join(data_dir, f"gt_mask_{year}.png")  # Update with your ground truth mask naming
    # gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    # plt.subplot(1, 3, 3)
    # plt.imshow(gt_mask, cmap='gray')
    # plt.title(f"Ground Truth Mask {year}")
    # plt.axis('off')

    plt.tight_layout()
    plt.show()

    years = ["2017", "2020", "2024"]
plt.figure(figsize=(15, 5))

for i, year in enumerate(years):
    mask = cv2.imread(os.path.join(output_dir, f"mask_{year}.png"), cv2.IMREAD_GRAYSCALE)

    plt.subplot(1, 3, i + 1)
    plt.imshow(mask, cmap="gray")
    plt.title(f"Vegetation Mask - {year}")

plt.show()

