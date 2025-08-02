import os
import random
import shutil

# Configuration
source_dir = '../deepglobe/train'  # Update if your folder path differs
train_img_dir = '../data/train/images'
train_mask_dir = '../data/train/masks'
valid_img_dir = '../data/valid/images'
valid_mask_dir = '../data/valid/masks'
num_images = 100
train_split = 80
valid_split = 20

# Create directories
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_mask_dir, exist_ok=True)
os.makedirs(valid_img_dir, exist_ok=True)
os.makedirs(valid_mask_dir, exist_ok=True)

# List all images
images = [f for f in os.listdir(source_dir) if f.endswith('_sat.jpg')]
if len(images) < num_images:
    print(f'Error: Only {len(images)} images found, need {num_images}.')
    exit(1)
print(f'Found {len(images)} images in {source_dir}')

# Select 50 random images
selected = random.sample(images, num_images)
train_imgs = selected[:train_split]
valid_imgs = selected[train_split:]

# Copy training files
print('Copying training images and masks...')
for img in train_imgs:
    shutil.copy(f'{source_dir}/{img}', f'{train_img_dir}/{img}')
    mask = img.replace('_sat.jpg', '_mask.png')
    if not os.path.exists(f'{source_dir}/{mask}'):
        print(f'Error: Mask {mask} not found for {img}')
        continue
    shutil.copy(f'{source_dir}/{mask}', f'{train_mask_dir}/{mask}')

# Copy validation files
print('Copying validation images and masks...')
for img in valid_imgs:
    shutil.copy(f'{source_dir}/{img}', f'{valid_img_dir}/{img}')
    mask = img.replace('_sat.jpg', '_mask.png')
    if not os.path.exists(f'{source_dir}/{mask}'):
        print(f'Error: Mask {mask} not found for {img}')
        continue
    shutil.copy(f'{source_dir}/{mask}', f'{valid_mask_dir}/{mask}')

print(f'Selected {num_images} images and masks. Files are in {train_img_dir}, {train_mask_dir}, {valid_img_dir}, and {valid_mask_dir}.')