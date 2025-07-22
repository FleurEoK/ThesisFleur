import numpy as np
from PIL import Image
import random
import json
import os

def preprocess_image(image_path, permutation, output_path):
    """
    Divide image into 16x16 patches and rearrange them according to permutation.
    
    Args:
        image_path (str): Path to input image
        permutation (list): List of integers representing new order of patches
        output_path (str): Path to save preprocessed image
    """
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Get image dimensions
    height, width = img_array.shape[:2]
    
    # Calculate number of patches
    patch_size = 16
    patches_per_row = width // patch_size
    patches_per_col = height // patch_size
    total_patches = patches_per_row * patches_per_col
    
    # Extract patches
    patches = []
    for row in range(patches_per_col):
        for col in range(patches_per_row):
            y_start = row * patch_size
            y_end = y_start + patch_size
            x_start = col * patch_size
            x_end = x_start + patch_size
            
            patch = img_array[y_start:y_end, x_start:x_end]
            patches.append(patch)
    
    # Apply permutation
    permuted_patches = [patches[i] for i in permutation]
    
    # Reconstruct image with permuted patches
    reconstructed = np.zeros_like(img_array)
    
    for idx, patch in enumerate(permuted_patches):
        row = idx // patches_per_row
        col = idx % patches_per_row
        
        y_start = row * patch_size
        y_end = y_start + patch_size
        x_start = col * patch_size
        x_end = x_start + patch_size
        
        reconstructed[y_start:y_end, x_start:x_end] = patch
    
    # Save preprocessed image
    result_img = Image.fromarray(reconstructed.astype(np.uint8))
    result_img.save(output_path)
    
    return permutation

def postprocess_image(image_path, permutation, output_path):
    """
    Reverse the permutation to reconstruct original image.
    
    Args:
        image_path (str): Path to preprocessed image
        permutation (list): Original permutation used in preprocessing
        output_path (str): Path to save reconstructed image
    """
    # Create inverse permutation
    inverse_permutation = [0] * len(permutation)
    for i, val in enumerate(permutation):
        inverse_permutation[val] = i
    
    # Apply inverse permutation
    preprocess_image(image_path, inverse_permutation, output_path)

def generate_random_permutation(num_patches):
    """Generate a random permutation for the given number of patches."""
    permutation = list(range(num_patches))
    random.shuffle(permutation)
    return permutation

if __name__ == "__main__":
    # Image parameters
    image_path = "cat.jpg"
    patch_size = 16
    num_permutations = 16
    
    # Create output directory
    output_dir = "permuted_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image to get dimensions
    img = Image.open(image_path)
    width, height = img.size
    
    # Calculate number of patches
    patches_per_row = width // patch_size
    patches_per_col = height // patch_size
    total_patches = patches_per_row * patches_per_col
    
    print(f"Image dimensions: {width}x{height}")
    print(f"Number of patches: {total_patches} ({patches_per_row}x{patches_per_col})")
    print(f"Generating {num_permutations} permutations...")
    
    # Generate 16 different permutations and process images
    permutation_data = {}
    
    for i in range(num_permutations):
        # Generate random permutation
        permutation = generate_random_permutation(total_patches)
        
        # Create output filename
        output_filename = f"permuted_{i:02d}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        # Preprocess image with this permutation
        print(f"Processing permutation {i+1}/{num_permutations}...")
        preprocess_image(image_path, permutation, output_path)
        
        # Store permutation data
        permutation_data[output_filename] = {
            "permutation_id": i,
            "permutation": permutation,
            "output_path": output_path
        }
    
    # Save permutation data to JSON
    json_path = os.path.join(output_dir, "permutations.json")
    with open(json_path, 'w') as f:
        json.dump(permutation_data, f, indent=2)
    
    print(f"Done! Generated {num_permutations} permuted images in '{output_dir}/' directory")
    print(f"Permutation data saved to '{json_path}'")