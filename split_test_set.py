import os
import random
import shutil

random.seed(42)

val_path = '/home/ray/proj/IDL_11785_Project/data/imagenet100/val_data'
test_path = '/home/ray/proj/IDL_11785_Project/data/imagenet100/test_data'
NUM_IMAGES = 20

os.makedirs(test_path, exist_ok=True)

for class_name in os.listdir(val_path):
    val_class_path = os.path.join(val_path, class_name)

    # Only process directories
    if not os.path.isdir(val_class_path):
        continue

    # Create the corresponding class directory in the destination
    test_class_path = os.path.join(test_path, class_name)
    os.makedirs(test_class_path, exist_ok=True)

    # Get all images in the current class directory
    val_images = [f for f in os.listdir(val_class_path) if os.path.isfile(os.path.join(val_class_path, f))]

    # Select a deterministic subset of images
    random.seed(42)  # Fixed seed for reproducibility
    test_images = random.sample(val_images, NUM_IMAGES)

    # Copy selected images to the new directory
    for image in test_images:
        val_class_image_path = os.path.join(val_class_path, image)
        test_image_path = os.path.join(test_class_path, image)
        shutil.copy(val_class_image_path, test_class_path)

    print(f"Copied {len(test_images)} images for class {class_name}.")

print('Done Splitting Test Set')
print('Checking if test and val have the same classes...')
test_dirs = os.listdir(test_path)
val_dirs = os.listdir(val_path)
for class_name in val_dirs:
    if class_name not in test_dirs:
        raise Exception(f'class {class_name} in validation set is not in the test set!')
print("Success!")