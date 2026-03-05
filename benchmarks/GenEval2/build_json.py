# According to the evaluation requirement, a json file with prompt as key and image path as value is needed.

import os
import json
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, required=True, help='Directory containing generated images.')
args = parser.parse_args()

image_dir = args.image_dir
image_paths = {}
for filepath in glob.iglob(os.path.join(image_dir, '**', '*.png'), recursive=True):
    filename = os.path.basename(filepath)
    image_key = filename.replace(".png", "")
    image_paths[image_key] = filepath
json_path = os.path.join(image_dir, 'image_paths.json')
with open(json_path, 'w') as f:
    json.dump(image_paths, f, indent=4)
print(f'Image paths json saved to {json_path}')