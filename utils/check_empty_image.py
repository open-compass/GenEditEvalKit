#!/usr/bin/env python3
import os
import sys
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm

def is_blank_image(image_path, threshold=0.98):
    """
    Check if an image is blank (mostly white).
    threshold: Pixel similarity threshold, default is 98% of pixels being the same to consider it blank.
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB mode
            img_rgb = img.convert('RGB')
            img_array = np.array(img_rgb)
            
            # Check for pure white (255, 255, 255)
            white_pixels = np.all(img_array == [255, 255, 255], axis=2)
            white_ratio = np.sum(white_pixels) / (img_array.shape[0] * img_array.shape[1])
            
            return white_ratio >= threshold
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def check_blank_images(directory):
    """Check all image files in the directory for blank images."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    blank_images = []
    
    # First collect all image file paths
    all_images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_path = os.path.join(root, file)
                all_images.append(image_path)
    
    total_images = len(all_images)
    
    # Use tqdm to show progress
    for image_path in tqdm(all_images, desc="Checking images", unit="img"):
        if is_blank_image(image_path):
            blank_images.append(image_path)
            tqdm.write(f"Blank image found: {image_path}")  # Use tqdm.write to avoid breaking the progress bar
    
    print(f"\nSummary:")
    print(f"Total images: {total_images}")
    print(f"Blank images: {len(blank_images)}")
    print(f"Blank ratio: {len(blank_images)/total_images*100:.2f}%" if total_images > 0 else "No images found")
    
    return len(blank_images) > 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_blank_images.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        sys.exit(1)
    
    has_blank = check_blank_images(directory)
    sys.exit(1 if has_blank else 0)  # Return 1 if blank images are found, else 0