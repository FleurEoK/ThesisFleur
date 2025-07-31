import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from skimage import morphology
from scipy import ndimage

def detect_dashed_rectangles(image_path, debug=False):
    """
    Detect dashed line rectangles in an image.
    
    Args:
        image_path: Path to the input image
        debug: Whether to show debug information
    
    Returns:
        List of rectangle coordinates [(x, y, w, h), ...]
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to different color spaces to better detect red dashed lines
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Create mask for red colors (dashed lines are typically red)
    # HSV ranges for red
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Also try to detect based on BGR values for red
    bgr_lower = np.array([0, 0, 100])  # Lower bound for red in BGR
    bgr_upper = np.array([100, 100, 255])  # Upper bound for red in BGR
    bgr_mask = cv2.inRange(image, bgr_lower, bgr_upper)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(red_mask, bgr_mask)
    
    # Morphological operations to connect dashed lines
    kernel = np.ones((3, 3), np.uint8)
    
    # Dilate to connect nearby dash segments
    dilated = cv2.dilate(combined_mask, kernel, iterations=2)
    
    # Create separate kernels for horizontal and vertical line detection
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    
    # Detect horizontal and vertical lines separately
    horizontal_lines = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, vertical_kernel)
    
    # Combine horizontal and vertical lines
    lines_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
    
    # Further dilate to connect line segments into rectangles
    lines_mask = cv2.dilate(lines_mask, kernel, iterations=3)
    
    if debug:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original')
        plt.subplot(1, 3, 2)
        plt.imshow(combined_mask, cmap='gray')
        plt.title('Red Detection')
        plt.subplot(1, 3, 3)
        plt.imshow(lines_mask, cmap='gray')
        plt.title('Line Detection')
        plt.show()
    
    # Find contours of the detected rectangles
    contours, _ = cv2.findContours(lines_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangles = []
    min_area = 500  # Minimum area for a valid rectangle
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter based on aspect ratio and size
        aspect_ratio = w / h
        if 0.1 < aspect_ratio < 10:  # Reasonable aspect ratio
            rectangles.append((x, y, w, h))
    
    return rectangles, image

def alternative_line_detection(image_path, debug=False):
    """
    Alternative method using Hough line detection for dashed rectangles.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                           minLineLength=20, maxLineGap=10)
    
    rectangles = []
    if lines is not None:
        # Group lines into potential rectangles
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Classify as horizontal or vertical
            if abs(y2 - y1) < abs(x2 - x1):  # More horizontal
                horizontal_lines.append((min(x1, x2), max(x1, x2), (y1 + y2) // 2))
            else:  # More vertical
                vertical_lines.append((min(y1, y2), max(y1, y2), (x1 + x2) // 2))
        
        # Try to form rectangles from line intersections
        # This is a simplified approach - you might need more sophisticated clustering
        for h_line in horizontal_lines:
            for v_line in vertical_lines:
                # Check if lines could form a rectangle corner
                h_x1, h_x2, h_y = h_line
                v_y1, v_y2, v_x = v_line
                
                # If lines intersect approximately
                if (h_x1 <= v_x <= h_x2) and (v_y1 <= h_y <= v_y2):
                    # This could be a rectangle corner
                    rectangles.append((min(h_x1, v_x), min(v_y1, h_y), 
                                    abs(h_x2 - h_x1), abs(v_y2 - v_y1)))
    
    return rectangles, image

def create_heatmap_from_rectangles(image_shape, rectangles):
    """
    Create a heatmap from rectangle coordinates.
    
    Args:
        image_shape: Shape of the original image (height, width, channels)
        rectangles: List of (x, y, w, h) tuples
    
    Returns:
        2D numpy array representing the heatmap
    """
    height, width = image_shape[:2]
    heatmap = np.zeros((height, width), dtype=np.int32)
    
    print(f"Creating heatmap for image shape: {height}x{width}")
    print(f"Processing {len(rectangles)} rectangles:")
    
    for i, (x, y, w, h) in enumerate(rectangles):
        print(f"  Rectangle {i+1}: x={x}, y={y}, w={w}, h={h}")
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x, width - 1))
        y1 = max(0, min(y, height - 1))
        x2 = max(0, min(x + w, width))
        y2 = max(0, min(y + h, height))
        
        print(f"    Clipped to: ({x1},{y1}) to ({x2},{y2})")
        
        if x2 > x1 and y2 > y1:
            heatmap[y1:y2, x1:x2] += 1
            pixels_added = (y2 - y1) * (x2 - x1)
            print(f"    Added {pixels_added} pixels to heatmap")
        else:
            print(f"    Invalid rectangle - skipped")
    
    print(f"Heatmap stats: min={heatmap.min()}, max={heatmap.max()}, non-zero pixels={np.count_nonzero(heatmap)}")
    return heatmap

def process_single_image(image_path, output_dir, detection_method='color', show_debug=False):
    """
    Process a single image to create its heatmap.
    """
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\nProcessing: {base_name}")
    
    try:
        # Try different detection methods
        if detection_method == 'color':
            rectangles, original_image = detect_dashed_rectangles(image_path, debug=show_debug)
        else:
            rectangles, original_image = alternative_line_detection(image_path, debug=show_debug)
        
        print(f"Detected {len(rectangles)} rectangles")
        
        if len(rectangles) == 0:
            print("No rectangles detected, trying alternative method...")
            rectangles, original_image = alternative_line_detection(image_path, debug=show_debug)
            print(f"Alternative method found {len(rectangles)} rectangles")
        
        # Create heatmap
        heatmap = create_heatmap_from_rectangles(original_image.shape, rectangles)
        
        # Visualize results
        plt.figure(figsize=(15, 5))
        
        # Original image with detected rectangles
        plt.subplot(1, 3, 1)
        display_image = original_image.copy()
        for x, y, w, h in rectangles:
            cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected Rectangles ({len(rectangles)})")
        plt.axis('off')
        
        # Original image
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')
        
        # Heatmap
        plt.subplot(1, 3, 3)
        if heatmap.max() > 0:
            im = plt.imshow(heatmap, cmap='hot', interpolation='nearest')
            plt.colorbar(im, label='Region Overlap Count')
        else:
            plt.imshow(np.zeros_like(heatmap), cmap='gray')
            plt.text(heatmap.shape[1]//2, heatmap.shape[0]//2, 'No regions detected', 
                    ha='center', va='center', fontsize=12, color='white')
        plt.title(f"Heatmap (Max: {heatmap.max()})")
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save results
        viz_path = os.path.join(output_dir, f"{base_name}_analysis.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save raw heatmap
        if heatmap.max() > 0:
            heatmap_path = os.path.join(output_dir, f"{base_name}_heatmap.npy")
            np.save(heatmap_path, heatmap)
        
        return heatmap
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def process_image_folder(input_folder, output_folder, detection_method='color', show_debug=False):
    """
    Process all JPEG images in a folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all JPEG images
    image_extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
    
    if not image_paths:
        print(f"No JPEG images found in {input_folder}")
        return
    
    print(f"Found {len(image_paths)} images to process")
    
    successful = 0
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n{'='*50}")
        print(f"Processing {i}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        heatmap = process_single_image(
            image_path, 
            output_folder, 
            detection_method=detection_method,
            show_debug=show_debug
        )
        
        if heatmap is not None:
            successful += 1
            max_overlap = heatmap.max()
            print(f"SUCCESS: Max overlap = {max_overlap}")
        else:
            print("FAILED")
    
    print(f"\nCompleted! Successfully processed {successful}/{len(image_paths)} images")

# Example usage
if __name__ == "__main__":
    # Configuration
    INPUT_FOLDER = "C:/Users/ensin/OneDrive/Documenten/Universiteit/Thesis/MasterThesis/ThesisFleur/Falcon/FALcon-main/results/imagenet_images"  # Change this
    OUTPUT_FOLDER = "heatmap_results_color4"
    DETECTION_METHOD = 'color'  # 'color' or 'lines'
    SHOW_DEBUG = False
    
    # Process single image for testing
    # process_single_image("your_test_image.jpg", OUTPUT_FOLDER, 
    #                     detection_method=DETECTION_METHOD, show_debug=SHOW_DEBUG)
    
    # Process all images
    process_image_folder(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        detection_method=DETECTION_METHOD,
        show_debug=SHOW_DEBUG
    )