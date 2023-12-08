import os
from PIL import Image

# Define the source and destination directories
source_directory = './Fake_ours_zo2m_mdeit_cifar10_thres_0.98_lr_0.01_tv_1.3e_3_5seed'
destination_directory = './Fake_ours_zo2m_mdeit_cifar10_thres_0.98_lr_0.01_tv_1.3e_3_5seed_32'

# Function to process and save the images
def process_and_save_images(src_dir, dst_dir):
    for subdir in os.listdir(src_dir):
        class_path = os.path.join(src_dir, subdir)
        if os.path.isdir(class_path):
            # Create a corresponding subdirectory in the destination directory
            destination_subdir = os.path.join(dst_dir, subdir)
            os.makedirs(destination_subdir, exist_ok=True)
            
            # Process each image in the subdirectory
            for filename in os.listdir(class_path):
                file_path = os.path.join(class_path, filename)
                with Image.open(file_path) as img:
                    # Resize the image
                    img_resized = img.resize((32, 32), Image.Resampling.LANCZOS)
                    
                    # Save the resized image to the corresponding subdirectory
                    img_resized.save(os.path.join(destination_subdir, filename))

# Create the destination directory
os.makedirs(destination_directory, exist_ok=True)

# Process and save images
process_and_save_images(source_directory, destination_directory)

print("All images have been resized and saved to the destination directory.")
