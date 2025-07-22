import json
import os
import argparse
import numpy as np
from PIL import Image

def load_permutation_data(json_path):
    """Load the permutation data from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def simple_deshuffle_approach(image, permutation):
    """
    Simple approach: just show the reconstructed images without deshuffling
    OR try a different deshuffling strategy
    """
    img_array = np.array(image)
    h, w, c = img_array.shape
    
    print(f"Image size: {h}x{w}")
    print(f"Permutation length: {len(permutation)}")
    
    # For now, let's NOT deshuffle and just return the image as-is
    # This will show how well the model reconstructed the permuted images
    return image

def try_pixel_level_deshuffle(image, permutation, target_size=64):
    """
    Try to work with pixel-level permutation by resizing image to match permutation
    """
    img_array = np.array(image)
    
    # Resize image to match permutation grid (64x64)
    if len(permutation) == 4096:  # 64x64
        resized_img = image.resize((64, 64), Image.LANCZOS)
        resized_array = np.array(resized_img)
        
        print(f"Resized image to 64x64 to match permutation")
        
        # Create reverse permutation
        reverse_perm = np.zeros(len(permutation), dtype=int)
        for new_pos, original_pos in enumerate(permutation):
            if original_pos < len(reverse_perm):
                reverse_perm[original_pos] = new_pos
        
        # Flatten image for pixel-level shuffling
        flat_img = resized_array.reshape(-1, 3)  # (4096, 3)
        
        # Apply reverse permutation
        deshuffled_flat = np.zeros_like(flat_img)
        for i in range(min(len(flat_img), len(reverse_perm))):
            if reverse_perm[i] < len(flat_img):
                deshuffled_flat[i] = flat_img[reverse_perm[i]]
        
        # Reshape back to image
        deshuffled_64 = deshuffled_flat.reshape(64, 64, 3)
        
        # Resize back to 224x224
        deshuffled_img = Image.fromarray(deshuffled_64.astype(np.uint8))
        final_img = deshuffled_img.resize((224, 224), Image.LANCZOS)
        
        return final_img
    else:
        return image

def main():
    parser = argparse.ArgumentParser('Simple Image Processing')
    parser.add_argument('--json_path', required=True, help='Path to JSON file with permutation data')
    parser.add_argument('--reconstructed_dir', default='C:/Users/ensin/OneDrive/Documenten/Universiteit/Thesis/SemAIM/reconstruction_analysis', help='Directory with reconstructed images')
    parser.add_argument('--output_dir', default='./simple_output', help='Output directory')
    parser.add_argument('--mode', default='no_deshuffle', choices=['no_deshuffle', 'pixel_level'], 
                       help='Processing mode: no_deshuffle (just copy) or pixel_level (try pixel deshuffle)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load permutation data
    print("Loading permutation data...")
    permutation_data = load_permutation_data(args.json_path)
    
    print(f"🔧 Mode: {args.mode}")
    
    # Process each image
    processed_count = 0
    for perm_key, perm_info in permutation_data.items():
        print(f"\n📷 Processing {perm_key}...")
        
        # Get the base filename
        base_filename = perm_key.replace('.jpg', '').replace('.png', '')
        
        # Look for reconstructed image
        reconstructed_path = os.path.join(args.reconstructed_dir, f"{base_filename}_reconstruction.png")
        
        if not os.path.exists(reconstructed_path):
            print(f"  ❌ Reconstructed image not found: {reconstructed_path}")
            continue
        
        try:
            # Load reconstructed image
            reconstructed_img = Image.open(reconstructed_path).convert('RGB')
            print(f"  ✅ Loaded reconstructed image: {reconstructed_img.size}")
            
            # Get permutation
            permutation = perm_info['permutation']
            
            # Process based on mode
            if args.mode == 'no_deshuffle':
                print("  ➡️  Not deshuffling - showing permuted reconstruction")
                processed_img = reconstructed_img
                output_suffix = "_permuted_reconstruction"
                
            elif args.mode == 'pixel_level':
                print("  🔄 Trying pixel-level deshuffling...")
                processed_img = try_pixel_level_deshuffle(reconstructed_img, permutation)
                output_suffix = "_pixel_deshuffled"
            
            # Save processed image
            output_path = os.path.join(args.output_dir, f"{base_filename}{output_suffix}.png")
            processed_img.save(output_path)
            
            print(f"  ✅ Saved: {output_path}")
            processed_count += 1
            
        except Exception as e:
            print(f"  ❌ Error processing {perm_key}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n🎉 Successfully processed {processed_count} images!")
    print(f"Results saved in: {args.output_dir}")
    
    if args.mode == 'no_deshuffle':
        print("\n📋 These images show the RECONSTRUCTED PERMUTED images (as the model learned them)")
        print("   Compare these to the original permuted images to see reconstruction quality")
    elif args.mode == 'pixel_level':
        print("\n📋 These images show PIXEL-LEVEL DESHUFFLED reconstructions")
        print("   These should look more like the original cat (if deshuffling worked correctly)")

if __name__ == '__main__':
    main()