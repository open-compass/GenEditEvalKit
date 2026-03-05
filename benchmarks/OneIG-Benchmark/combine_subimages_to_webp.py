import os
import argparse
from PIL import Image
import glob
from collections import defaultdict
from tqdm import tqdm

class_item = {
    "Anime_Stylization" : "anime",
    "Portrait" : "human",
    "General_Object" : "object",
    "Text_Rendering" : "text",
    "Knowledge_Reasoning" : "reasoning",
    "Multilingualism" : "multilingualism"
}

def create_image_gallery(images, rows=2, cols=2): # Stitch multiple images into a single gallery image. Input is a list of PIL.Image.
    assert len(images) >= rows * cols, "Not enough images provided!"
    img_height, img_width = images[0].size
    # Create a blank image as the gallery background
    gallery_width = cols * img_width
    gallery_height = rows * img_height
    gallery_image = Image.new("RGB", (gallery_width, gallery_height))
    # Paste each image onto the gallery canvas
    for row in range(rows):
        for col in range(cols):
            img = images[row * cols + col]  # Convert numpy array to PIL image
            x_offset = col * img_width
            y_offset = row * img_height
            gallery_image.paste(img, (x_offset, y_offset))
    return gallery_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subimage_dir', type=str, required=True, help='Path to the folder containing sub-images.')
    args = parser.parse_args()

    for item_class in class_item.values():
        subimage_dir = f"{args.subimage_dir}/{item_class}"

        # Group images under the same class by item['id']
        png_files = glob.glob(os.path.join(subimage_dir, "**", "*.png"))
        groups = defaultdict(list)
        for f in png_files:
            fname = os.path.basename(f)
            if "_" in fname:
                item_id = fname.split("_")[0]
                groups[item_id].append(f)

        # Combine each group of 4 images
        for item_id, files in tqdm(groups.items()):
            assert len(files) == 4, f"Each group should contain 4 images; group {item_id} contains {len(files)} images"
            # Create gallery image
            files = sorted(files)
            images = [Image.open(f) for f in files]
            gallery = create_image_gallery(images, rows=2, cols=2)
            # Save image
            assert "images_before_combination" in files[0] and "_0.png" in files[0], f"File path format incorrect; does not contain 'images_before_combination' and '_0.png': {files[0]}" # Use the first file path to construct output path, also verify order
            save_path = files[0].replace("images_before_combination", "images").replace("_0.png", ".webp")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            gallery.save(save_path, "WEBP")


# First combine images, save to a new folder, then check for errors
if __name__ == '__main__':
    main()