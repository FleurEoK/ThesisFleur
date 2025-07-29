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
    Precisely detect dashed line rectangles in an image, avoiding false positives.
    
    Args:
        image_path: Path to the input image
        debug: Whether to show debug information
    
    Returns:
        Tuple: (rectangles, image) where rectangles is [(x, y, w, h), ...]
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to HSV for better red detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # More precise red detection for dashed lines
    # Focus on bright, saturated reds typical of dashed lines
    lower_red1 = np.array([0, 120, 70])     # Higher saturation and value
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])   # Higher saturation and value
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # More restrictive BGR detection
    bgr_lower = np.array([0, 0, 150])    # Higher red threshold
    bgr_upper = np.array([80, 80, 255])  # Lower blue/green threshold
    bgr_mask = cv2.inRange(image, bgr_lower, bgr_upper)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(red_mask, bgr_mask)
    
    # Remove small noise first
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, noise_kernel)
    
    if debug:
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(combined_mask, cmap='gray')
        plt.title('Refined Red Detection')
        plt.axis('off')
    
    # CONSERVATIVE morphological operations - only connect obvious dashed lines
    # Use smaller kernels and fewer iterations
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 2))  # Smaller
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 12))    # Smaller
    
    # Connect dashes more conservatively
    horizontal_lines = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, horizontal_kernel, iterations=1)
    vertical_lines = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)
    
    # Combine lines
    lines_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
    
    if debug:
        plt.subplot(1, 4, 3)
        plt.imshow(lines_mask, cmap='gray')
        plt.title('Connected Lines')
        plt.axis('off')
    
    # Very conservative dilation to form rectangles
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Much smaller
    lines_mask = cv2.dilate(lines_mask, connect_kernel, iterations=1)  # Only 1 iteration
    
    # Clean up small artifacts
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    lines_mask = cv2.morphologyEx(lines_mask, cv2.MORPH_OPEN, clean_kernel)
    
    if debug:
        plt.subplot(1, 4, 4)
        plt.imshow(lines_mask, cmap='gray')
        plt.title('Final Detection')
        plt.axis('off')
        plt.show()
    
    # Find contours
    contours, _ = cv2.findContours(lines_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangles = []
    min_area = 5000        # Much higher minimum area (70x70 pixels)
    max_area = image.shape[0] * image.shape[1] * 0.5  # Max 50% of image
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < min_area or area > max_area:
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Strict validation for rectangle-like properties
        aspect_ratio = w / h
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate how rectangular the shape is
        rect_area = w * h
        extent = area / rect_area  # How much of bounding rect is filled
        
        # Calculate solidity (area vs convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Very strict filtering
        if (0.2 < aspect_ratio < 5 and      # Reasonable aspect ratio
            w > 50 and h > 50 and           # Minimum size
            extent > 0.3 and                # Must fill reasonable portion of bounding rect
            solidity > 0.7 and              # Must be reasonably solid
            perimeter > 200):               # Must have substantial perimeter
            
            rectangles.append((x, y, w, h))
    
    # Additional filtering: remove rectangles that are too close or overlapping
    rectangles = filter_overlapping_rectangles(rectangles)
    
    # Final validation: check if detected rectangles actually look like dashed rectangles
    rectangles = validate_dashed_rectangles(image, rectangles, debug)
    
    return rectangles, image

def filter_overlapping_rectangles(rectangles, min_distance=30):
    """
    Remove rectangles that are too close to each other or overlapping.
    """
    if len(rectangles) <= 1:
        return rectangles
    
    # Sort by area (keep larger ones)
    rectangles = sorted(rectangles, key=lambda r: r[2] * r[3], reverse=True)
    
    filtered = []
    for rect in rectangles:
        x1, y1, w1, h1 = rect
        
        # Check if this rectangle is too close to any existing one
        too_close = False
        for existing in filtered:
            x2, y2, w2, h2 = existing
            
            # Calculate distance between centers
            center1 = (x1 + w1/2, y1 + h1/2)
            center2 = (x2 + w2/2, y2 + h2/2)
            distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            
            # Check overlap
            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = overlap_x * overlap_y
            
            if distance < min_distance or overlap_area > 0:
                too_close = True
                break
        
        if not too_close:
            filtered.append(rect)
    
    return filtered

def validate_dashed_rectangles(image, rectangles, debug=False):
    """
    Final validation to ensure detected rectangles actually contain dashed lines.
    """
    validated = []
    
    for i, (x, y, w, h) in enumerate(rectangles):
        # Extract the region
        roi = image[y:y+h, x:x+w]
        
        # Check if this region actually contains dashed line patterns
        if is_dashed_rectangle_region(roi, debug):
            validated.append((x, y, w, h))
        elif debug:
            print(f"Rectangle {i+1} failed dashed line validation")
    
    return validated

def is_dashed_rectangle_region(roi, debug=False):
    """
    Check if a region contains dashed line patterns on the edges.
    """
    if roi.shape[0] < 50 or roi.shape[1] < 50:
        return False
    
    # Convert to HSV
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Detect red in the ROI
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Check edges for red pixels (dashed lines should be on the perimeter)
    h, w = red_mask.shape
    edge_width = 5  # Check 5 pixels from each edge
    
    # Extract edge regions
    top_edge = red_mask[:edge_width, :]
    bottom_edge = red_mask[-edge_width:, :]
    left_edge = red_mask[:, :edge_width]
    right_edge = red_mask[:, -edge_width:]
    
    # Count red pixels on edges
    edge_red_pixels = (np.sum(top_edge > 0) + np.sum(bottom_edge > 0) + 
                      np.sum(left_edge > 0) + np.sum(right_edge > 0))
    
    total_edge_pixels = (top_edge.size + bottom_edge.size + 
                        left_edge.size + right_edge.size)
    
    edge_red_ratio = edge_red_pixels / total_edge_pixels
    
    # Should have some red on edges (dashed lines) but not too much (solid red)
    return 0.02 < edge_red_ratio < 0.4

def alternative_line_detection(image_path, debug=False):
    """
    Alternative method using Hough line detection for dashed rectangles.
    """
    image = cv2.imread(image_path)
    
    # Focus on red channel for better line detection
    red_channel = image[:, :, 2]
    
    # Apply threshold to get bright red areas
    _, thresh = cv2.threshold(red_channel, 150, 255, cv2.THRESH_BINARY)
    
    # Apply edge detection
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    
    # Detect lines using Hough transform with more restrictive parameters
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,  # Higher threshold
                           minLineLength=40, maxLineGap=20)     # Longer lines, smaller gaps
    
    rectangles = []
    if lines is not None and len(lines) >= 4:  # Need at least 4 lines for a rectangle
        # Group lines into potential rectangles
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # Only consider reasonably long lines
            if length < 30:
                continue
            
            # Classify as horizontal or vertical
            angle = np.abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            
            if angle < 15 or angle > 165:  # Horizontal (near 0 or 180 degrees)
                horizontal_lines.append((min(x1, x2), max(x1, x2), (y1 + y2) // 2))
            elif 75 < angle < 105:  # Vertical (near 90 degrees)
                vertical_lines.append((min(y1, y2), max(y1, y2), (x1 + x2) // 2))
        
        # Only proceed if we have enough lines
        if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
            rectangles = form_rectangles_from_lines(horizontal_lines, vertical_lines, 
                                                  image.shape, min_size=50)
    
    return rectangles, image

def form_rectangles_from_lines(h_lines, v_lines, image_shape, min_size=50):
    """
    Form rectangles from detected horizontal and vertical lines.
    """
    height, width = image_shape[:2]
    rectangles = []
    
    # Try all combinations of 2 horizontal and 2 vertical lines
    for i, h1 in enumerate(h_lines):
        for j, h2 in enumerate(h_lines[i+1:], i+1):
            for k, v1 in enumerate(v_lines):
                for l, v2 in enumerate(v_lines[k+1:], k+1):
                    
                    # Get line positions
                    h1_start, h1_end, h1_y = h1
                    h2_start, h2_end, h2_y = h2
                    v1_start, v1_end, v1_x = v1
                    v2_start, v2_end, v2_x = v2
                    
                    # Check if lines can form a rectangle
                    if abs(h1_y - h2_y) < min_size or abs(v1_x - v2_x) < min_size:
                        continue
                    
                    # Calculate potential rectangle
                    x = min(v1_x, v2_x)
                    y = min(h1_y, h2_y)
                    w = abs(v2_x - v1_x)
                    h = abs(h2_y - h1_y)
                    
                    # Validate rectangle
                    if (w >= min_size and h >= min_size and 
                        x >= 0 and y >= 0 and 
                        x + w <= width and y + h <= height):
                        rectangles.append((x, y, w, h))
    
    return rectangles

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
    print(f"Processing {len(rectangles)} validated rectangles:")
    
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
        
        print(f"Detected {len(rectangles)} valid rectangles")
        
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
    INPUT_FOLDER = "C:/Users/ensin/OneDrive/Documenten/Universiteit/Thesis/MasterThesis/ThesisFleur/Falcon/FALcon-main/results/imagenet_images"
    OUTPUT_FOLDER = "heatmap_results_precise"
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