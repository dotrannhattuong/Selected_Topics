import os
from PIL import Image

input_folder = '/mnt/HDD1/tuong/images_HW3'
output_folder = '/mnt/HDD1/tuong/images_HW3'
os.makedirs(output_folder, exist_ok=True)

target_size = (512, 512)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        with Image.open(input_path) as img:
            resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
            resized_img.save(output_path)

print("Finished resizing images.")
