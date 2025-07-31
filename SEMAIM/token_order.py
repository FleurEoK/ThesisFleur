import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import json
from collections import defaultdict
import matplotlib.patches as patches

def create_token_order_from_heatmap(heatmap, patch_size=16):
    """
    Create token order based on heatmap values, treating image as patches/tokens.
    
    Args:
        heatmap: 2D numpy array with heatmap values
        patch_size: Size of each patch/token (default 16x16 like ViT)
    
    Returns:
        Dictionary with token ordering information
    """
    height, width = heatmap.shape
    
    # Calculate number of patches
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    
    # Crop heatmap to fit exact patch grid
    cropped_height = num_patches_h * patch_size
    cropped_width = num_patches_w * patch_size
    cropped_heatmap = heatmap[:cropped_height, :cropped_width]
    
    print(f"Original heatmap shape: {height}x{width}")
    print(f"Patch grid: {num_patches_h}x{num_patches_w} = {num_patches_h * num_patches_w} tokens")
    print(f"Cropped to: {cropped_height}x{cropped_width}")
    
    # Calculate average heatmap value for each patch
    patch_values = []
    patch_positions = []
    
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            y_start = i * patch_size
            y_end = (i + 1) * patch_size
            x_start = j * patch_size
            x_end = (j + 1) * patch_size
            
            patch = cropped_heatmap[y_start:y_end, x_start:x_end]
            avg_value = np.mean(patch)
            
            patch_values.append(avg_value)
            patch_positions.append({
                'patch_id': i * num_patches_w + j,
                'row': i,
                'col': j,
                'pixel_coords': {
                    'y_start': y_start,
                    'y_end': y_end,
                    'x_start': x_start,
                    'x_end': x_end
                },
                'avg_heatmap_value': float(avg_value)
            })
    
    # Create ordering: highest heatmap values first
    sorted_indices = np.argsort(patch_values)[::-1]
    
    # Create token order
    token_order = []
    for order_idx, patch_idx in enumerate(sorted_indices):
        patch_info = patch_positions[patch_idx].copy()
        patch_info['order'] = order_idx
        token_order.append(patch_info)
    
    # Group by heatmap value for statistics
    value_groups = defaultdict(list)
    for patch in patch_positions:
        value_groups[patch['avg_heatmap_value']].append(patch['patch_id'])
    
    # Print some statistics
    unique_values = len(value_groups)
    max_val = max(patch_values) if patch_values else 0
    min_val = min(patch_values) if patch_values else 0
    nonzero_patches = sum(1 for v in patch_values if v > 0)
    
    print(f"Heatmap statistics:")
    print(f"  - Value range: {min_val:.3f} to {max_val:.3f}")
    print(f"  - Unique values: {unique_values}")
    print(f"  - Non-zero patches: {nonzero_patches}/{len(patch_values)}")
    
    return {
        'image_shape': {'height': height, 'width': width},
        'patch_size': patch_size,
        'num_patches': {'height': num_patches_h, 'width': num_patches_w, 'total': len(patch_positions)},
        'cropped_shape': {'height': cropped_height, 'width': cropped_width},
        'token_order': token_order,
        'value_statistics': {
            'unique_values': unique_values,
            'max_value': float(max_val),
            'min_value': float(min_val),
            'nonzero_patches': nonzero_patches,
            'value_distribution': {str(k): len(v) for k, v in value_groups.items()}
        }
    }

def find_matching_files(heatmap_folder, image_folder=None):
    """
    Find matching heatmap (.npy) and image files.
    
    Args:
        heatmap_folder: Folder containing *_heatmap.npy files
        image_folder: Folder containing image files (if different from heatmap_folder)
    """
    # Find all .npy heatmap files
    heatmap_files = glob.glob(os.path.join(heatmap_folder, "*_heatmap.npy"))
    
    # If no separate image folder specified, look in heatmap folder
    if image_folder is None:
        image_folder = heatmap_folder
    
    matches = []
    for heatmap_path in heatmap_files:
        base_name = os.path.basename(heatmap_path).replace('_heatmap.npy', '')
        
        # Look for corresponding image file
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            potential_image = os.path.join(image_folder, base_name + ext)
            if os.path.exists(potential_image):
                image_path = potential_image
                break
        
        matches.append({
            'base_name': base_name,
            'heatmap_path': heatmap_path,
            'image_path': image_path
        })
    
    return matches

def visualize_token_order(image, heatmap, token_order_info, output_path, show_top_n=20):
    """
    Create comprehensive visualization showing the token order analysis.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    patch_size = token_order_info['patch_size']
    
    # 1. Original image
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 2. Heatmap
    im1 = axes[0, 1].imshow(heatmap, cmap='hot', interpolation='nearest')
    axes[0, 1].set_title(f'Heatmap (Max: {heatmap.max()})')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], label='Region Overlap Count', shrink=0.8)
    
    # 3. Token grid overlay on original image
    axes[0, 2].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f'Token Grid ({patch_size}x{patch_size})')
    
    # Draw grid lines
    height, width = heatmap.shape
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    
    for i in range(0, num_patches_h * patch_size, patch_size):
        axes[0, 2].axhline(i, color='white', linewidth=0.5, alpha=0.7)
    for j in range(0, num_patches_w * patch_size, patch_size):
        axes[0, 2].axvline(j, color='white', linewidth=0.5, alpha=0.7)
    axes[0, 2].axis('off')
    
    # 4. Top-N tokens highlighted
    axes[1, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'Top {show_top_n} Priority Tokens')
    
    colors = plt.cm.plasma(np.linspace(0, 1, show_top_n))
    
    for i, token in enumerate(token_order_info['token_order'][:show_top_n]):
        coords = token['pixel_coords']
        
        # Draw rectangle
        rect = patches.Rectangle(
            (coords['x_start'], coords['y_start']),
            coords['x_end'] - coords['x_start'],
            coords['y_end'] - coords['y_start'],
            linewidth=2, 
            edgecolor=colors[i], 
            facecolor=colors[i],
            alpha=0.3
        )
        axes[1, 0].add_patch(rect)
        
        # Add order number
        center_x = (coords['x_start'] + coords['x_end']) / 2
        center_y = (coords['y_start'] + coords['y_end']) / 2
        axes[1, 0].text(center_x, center_y, str(i), 
                      ha='center', va='center', fontsize=8, 
                      color='white', weight='bold',
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.8))
    
    axes[1, 0].axis('off')
    
    # 5. Patch-level heatmap
    patch_heatmap = np.zeros((num_patches_h, num_patches_w))
    for token in token_order_info['token_order']:
        row, col = token['row'], token['col']
        patch_heatmap[row, col] = token['avg_heatmap_value']
    
    im2 = axes[1, 1].imshow(patch_heatmap, cmap='hot', interpolation='nearest')
    axes[1, 1].set_title('Average Heatmap Value per Token')
    plt.colorbar(im2, ax=axes[1, 1], label='Average Value', shrink=0.8)
    
    # Add grid lines
    for i in range(num_patches_h + 1):
        axes[1, 1].axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.5)
    for j in range(num_patches_w + 1):
        axes[1, 1].axvline(j - 0.5, color='white', linewidth=0.5, alpha=0.5)
    
    # 6. Token order distribution
    values = [token['avg_heatmap_value'] for token in token_order_info['token_order']]
    axes[1, 2].hist(values, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 2].set_title('Distribution of Token Values')
    axes[1, 2].set_xlabel('Average Heatmap Value')
    axes[1, 2].set_ylabel('Number of Tokens')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_token_sequence_visualization(image, token_order_info, output_path, tokens_per_row=10, max_tokens=50):
    """
    Create a visualization showing the actual image patches in their priority order.
    """
    token_order = token_order_info['token_order'][:max_tokens]  # Limit for readability
    
    if not token_order:
        print("No tokens to visualize")
        return
    
    # Calculate grid size
    num_rows = (len(token_order) + tokens_per_row - 1) // tokens_per_row
    
    fig, axes = plt.subplots(num_rows, tokens_per_row, 
                            figsize=(tokens_per_row * 1.5, num_rows * 1.5))
    
    if num_rows == 1:
        axes = axes.reshape(1, -1) if tokens_per_row > 1 else [axes]
    elif tokens_per_row == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, token in enumerate(token_order):
        row = idx // tokens_per_row
        col = idx % tokens_per_row
        
        # Extract patch from original image
        coords = token['pixel_coords']
        patch = image[coords['y_start']:coords['y_end'], 
                     coords['x_start']:coords['x_end']]
        
        if patch.size > 0:
            axes[row, col].imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        else:
            axes[row, col].imshow(np.zeros((16, 16, 3)))
        
        axes[row, col].set_title(f"{idx}: {token['avg_heatmap_value']:.2f}", 
                                fontsize=9)
        axes[row, col].axis('off')
    
    # Hide unused subplots
    total_used = len(token_order)
    for idx in range(total_used, num_rows * tokens_per_row):
        row = idx // tokens_per_row
        col = idx % tokens_per_row
        axes[row, col].axis('off')
    
    plt.suptitle(f'Token Sequence (Top {len(token_order)} by Priority)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_image_in_token_order(image, token_order_info, base_name, dataset_folder, viz_folder=None, tokens_per_row=None):
    """
    Display the entire image rearranged in token priority order.
    This shows what the image looks like when viewed in the order the model would process tokens.
    
    Args:
        image: Original image
        token_order_info: Token ordering information
        base_name: Base name for the image (without extension)
        dataset_folder: Folder to save dataset images (just the rearranged image)
        viz_folder: Optional folder to save visualization comparisons
        tokens_per_row: Number of tokens per row in rearranged image
    
    Returns:
        rearranged_image: The rearranged image array
    """
    token_order = token_order_info['token_order']
    patch_size = token_order_info['patch_size']
    total_tokens = len(token_order)
    
    if not token_order:
        print("No tokens to rearrange")
        return None
    
    # Auto-calculate grid size if not specified
    if tokens_per_row is None:
        # Try to keep it roughly square-ish
        tokens_per_row = int(np.ceil(np.sqrt(total_tokens)))
    
    num_rows = (total_tokens + tokens_per_row - 1) // tokens_per_row
    
    # Create the rearranged image
    rearranged_height = num_rows * patch_size
    rearranged_width = tokens_per_row * patch_size
    rearranged_image = np.zeros((rearranged_height, rearranged_width, 3), dtype=np.uint8)
    
    print(f"Creating rearranged image: {total_tokens} tokens in {num_rows}x{tokens_per_row} grid")
    print(f"Output size: {rearranged_height}x{rearranged_width}")
    
    # Place each token in its new position based on priority order
    for idx, token in enumerate(token_order):
        # Calculate position in the rearranged grid
        new_row = idx // tokens_per_row
        new_col = idx % tokens_per_row
        
        # Calculate pixel coordinates in rearranged image
        new_y_start = new_row * patch_size
        new_y_end = new_y_start + patch_size
        new_x_start = new_col * patch_size
        new_x_end = new_x_start + patch_size
        
        # Extract patch from original image
        coords = token['pixel_coords']
        original_patch = image[coords['y_start']:coords['y_end'], 
                              coords['x_start']:coords['x_end']]
        
        # Place in rearranged image
        if original_patch.size > 0:
            # Ensure patch is the right size (handle edge cases)
            if original_patch.shape[:2] != (patch_size, patch_size):
                original_patch = cv2.resize(original_patch, (patch_size, patch_size))
            
            rearranged_image[new_y_start:new_y_end, new_x_start:new_x_end] = original_patch
    
    # Save the rearranged image for dataset use
    dataset_image_path = os.path.join(dataset_folder, f"{base_name}_priority_rearranged.jpg")
    cv2.imwrite(dataset_image_path, rearranged_image)
    print(f"Saved dataset image: {dataset_image_path}")
    
    # Create visualization comparison if viz_folder is specified
    if viz_folder is not None:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Rearranged image
        axes[1].imshow(cv2.cvtColor(rearranged_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Image in Token Priority Order\n({total_tokens} tokens, {tokens_per_row} per row)')
        axes[1].axis('off')
        
        # Add grid lines to show token boundaries
        for i in range(0, rearranged_height, patch_size):
            axes[1].axhline(i, color='white', linewidth=0.5, alpha=0.7)
        for j in range(0, rearranged_width, patch_size):
            axes[1].axvline(j, color='white', linewidth=0.5, alpha=0.7)
        
        # Token priority heatmap
        priority_map = np.full((num_rows, tokens_per_row), -1, dtype=float)
        
        for idx, token in enumerate(token_order):
            new_row = idx // tokens_per_row
            new_col = idx % tokens_per_row
            # Use inverse order so highest priority (order 0) shows as highest value
            priority_map[new_row, new_col] = total_tokens - idx
        
        # Mask unfilled positions
        priority_map_masked = np.ma.masked_where(priority_map == -1, priority_map)
        
        im = axes[2].imshow(priority_map_masked, cmap='viridis', interpolation='nearest')
        axes[2].set_title('Token Priority Map\n(Brighter = Higher Priority)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[2], shrink=0.8)
        cbar.set_label('Priority Rank')
        
        # Add grid
        for i in range(num_rows + 1):
            axes[2].axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.5)
        for j in range(tokens_per_row + 1):
            axes[2].axvline(j - 0.5, color='white', linewidth=0.5, alpha=0.5)
        
        # Add text annotations for first few tokens
        for idx in range(min(10, total_tokens)):  # Show first 10 token numbers
            new_row = idx // tokens_per_row
            new_col = idx % tokens_per_row
            axes[2].text(new_col, new_row, str(idx), 
                        ha='center', va='center', fontsize=8, 
                        color='white', weight='bold')
        
        plt.suptitle('Image Rearranged by Token Priority', fontsize=16)
        plt.tight_layout()
        
        viz_comparison_path = os.path.join(viz_folder, f"{base_name}_rearranged_comparison.png")
        plt.savefig(viz_comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization: {viz_comparison_path}")
    
    return rearranged_image

def create_dataset_info_file(dataset_folder, results_summary, patch_size):
    """
    Create an info file for the dataset describing the rearrangement process.
    """
    info = {
        "dataset_description": "Priority-rearranged images based on heatmap token ordering",
        "creation_method": "Images rearranged by patch priority derived from annotation overlap heatmaps",
        "patch_size": patch_size,
        "total_images": len(results_summary),
        "file_naming": "original_name_priority_rearranged.jpg",
        "creation_date": str(np.datetime64('now')),
        "image_statistics": {
            "successful_conversions": sum(1 for r in results_summary.values() if r.get('rearranged_successfully', False)),
            "failed_conversions": sum(1 for r in results_summary.values() if not r.get('rearranged_successfully', False))
        }
    }
    
    info_path = os.path.join(dataset_folder, "dataset_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Created dataset info file: {info_path}")
    return info_path

def process_heatmap_results_folder(results_folder, output_folder="token_analysis", image_folder=None, 
                                 patch_size=16, create_dataset=True, show_debug=False):
    """
    Process a folder containing heatmap results to create token orderings and optionally a dataset.
    
    Args:
        results_folder: Folder containing *_heatmap.npy files
        output_folder: Output folder for token analysis results
        image_folder: Folder containing original images (if different from results_folder)
        patch_size: Size of each token patch
        create_dataset: Whether to create a separate dataset folder with rearranged images
        show_debug: Whether to print detailed debug information
    """
    # Create main output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Create subfolders
    analysis_folder = os.path.join(output_folder, "analysis")
    os.makedirs(analysis_folder, exist_ok=True)
    
    if create_dataset:
        dataset_folder = os.path.join(output_folder, "priority_rearranged_dataset")
        os.makedirs(dataset_folder, exist_ok=True)
        visualizations_folder = os.path.join(output_folder, "visualizations")
        os.makedirs(visualizations_folder, exist_ok=True)
    else:
        dataset_folder = None
        visualizations_folder = None
    
    # Find matching heatmap and image files
    matches = find_matching_files(results_folder, image_folder)
    
    if not matches:
        print(f"No heatmap files found in {results_folder}")
        print("Expected files: *_heatmap.npy")
        return
    
    print(f"Found {len(matches)} heatmap files to process")
    if image_folder:
        print(f"Looking for images in: {image_folder}")
    else:
        print(f"Looking for images in same folder: {results_folder}")
    print(f"Using patch size: {patch_size}x{patch_size}")
    
    if create_dataset:
        print(f"Dataset images will be saved to: {dataset_folder}")
        print(f"Visualizations will be saved to: {visualizations_folder}")
    
    all_results = {}
    successful = 0
    dataset_summary = {}
    
    for i, match in enumerate(matches, 1):
        print(f"\n{'='*60}")
        print(f"Processing {i}/{len(matches)}: {match['base_name']}")
        
        try:
            # Load heatmap
            heatmap = np.load(match['heatmap_path'])
            print(f"Loaded heatmap: {heatmap.shape}, range: {heatmap.min()}-{heatmap.max()}")
            
            # Load image if available
            image = None
            if match['image_path']:
                image = cv2.imread(match['image_path'])
                if image is not None:
                    print(f"Loaded image: {image.shape}")
                else:
                    print(f"Warning: Could not load image {match['image_path']}")
            else:
                print("No matching image found")
            
            # Create token order
            token_order_info = create_token_order_from_heatmap(heatmap, patch_size)
            
            # Create result
            result = {
                'image_name': match['base_name'],
                'heatmap_path': match['heatmap_path'],
                'image_path': match['image_path'],
                'token_ordering': token_order_info,
                'rearranged_successfully': False
            }
            
            # Save individual JSON
            json_path = os.path.join(analysis_folder, f"{match['base_name']}_token_order.json")
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Create visualizations if image is available
            if image is not None:
                # Main analysis visualization
                viz_path = os.path.join(analysis_folder, f"{match['base_name']}_token_analysis.png")
                visualize_token_order(image, heatmap, token_order_info, viz_path)
                
                # Token sequence visualization (top tokens only)
                seq_path = os.path.join(analysis_folder, f"{match['base_name']}_token_sequence.png")
                create_token_sequence_visualization(image, token_order_info, seq_path)
                
                # Create rearranged image
                rearranged_image = create_image_in_token_order(
                    image, 
                    token_order_info, 
                    match['base_name'],
                    dataset_folder if create_dataset else analysis_folder,
                    visualizations_folder if create_dataset else None
                )
                
                if rearranged_image is not None:
                    result['rearranged_successfully'] = True
                    if create_dataset:
                        dataset_summary[match['base_name']] = {
                            'original_image': match['image_path'],
                            'rearranged_image': os.path.join(dataset_folder, f"{match['base_name']}_priority_rearranged.jpg"),
                            'original_shape': image.shape,
                            'rearranged_shape': rearranged_image.shape,
                            'total_tokens': token_order_info['num_patches']['total'],
                            'nonzero_tokens': token_order_info['value_statistics']['nonzero_patches']
                        }
                
                print(f"Created all outputs")
            else:
                print("Skipped visualizations (no image)")
            
            all_results[match['base_name']] = result
            successful += 1
            
            # Print summary
            stats = token_order_info['value_statistics']
            print(f"Total tokens: {token_order_info['num_patches']['total']}")
            print(f"Non-zero tokens: {stats['nonzero_patches']}")
            print(f"Value range: {stats['min_value']:.3f} - {stats['max_value']:.3f}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            if show_debug:
                import traceback
                traceback.print_exc()
    
    # Save combined results
    combined_json_path = os.path.join(analysis_folder, "all_token_orders.json")
    with open(combined_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create dataset info if dataset was created
    if create_dataset and dataset_summary:
        dataset_info_path = create_dataset_info_file(dataset_folder, dataset_summary, patch_size)
        
        # Save dataset summary
        dataset_summary_path = os.path.join(dataset_folder, "dataset_summary.json")
        with open(dataset_summary_path, 'w') as f:
            json.dump(dataset_summary, f, indent=2)
        print(f"Created dataset summary: {dataset_summary_path}")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"Successfully processed: {successful}/{len(matches)} files")
    print(f"Results saved to: {output_folder}")
    print(f"  - Analysis files: {analysis_folder}")
    if create_dataset:
        print(f"  - Dataset images: {dataset_folder}")
        print(f"  - Visualizations: {visualizations_folder}")
        print(f"  - Dataset contains: {len(dataset_summary)} rearranged images")
    print(f"Combined analysis JSON: {combined_json_path}")
    
    return all_results, dataset_summary if create_dataset else None

if __name__ == "__main__":
    # Configuration
    HEATMAP_RESULTS_FOLDER = "heatmap_results_color"  # Folder with your *_heatmap.npy files
    IMAGE_FOLDER = "C:/Users/ensin/OneDrive/Documenten/Universiteit/Thesis/MasterThesis/ThesisFleur/Falcon/FALcon-main/results/imagenet_images/"  # Folder with original images
    OUTPUT_FOLDER = "token_analysis"
    PATCH_SIZE = 16  # ViT-style tokens (16x16)
    CREATE_DATASET = True  # Whether to create a separate dataset folder
    
    # Process the heatmap results
    results, dataset_info = process_heatmap_results_folder(
        results_folder=HEATMAP_RESULTS_FOLDER,
        image_folder=IMAGE_FOLDER,
        output_folder=OUTPUT_FOLDER,
        patch_size=PATCH_SIZE,
        create_dataset=CREATE_DATASET,
        show_debug=True
    )
    
    print(f"\nProcessed {len(results)} images successfully!")
    
    if CREATE_DATASET and dataset_info:
        print(f"Created dataset with {len(dataset_info)} images in priority-rearranged format")
        print("Dataset is ready for use in machine learning pipelines!")
    
    # Example: Access specific result
    # if results:
    #     first_result = list(results.values())[0]
    #     print(f"\nExample - First 5 tokens for {first_result['image_name']}:")
    #     for i, token in enumerate(first_result['token_ordering']['token_order'][:5]):
    #         print(f"  {i}: Row {token['row']}, Col {token['col']}, Value: {token['avg_heatmap_value']:.3f}")