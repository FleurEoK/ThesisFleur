import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, box
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

class RectangleOverlapAnalyzer:
    def __init__(self, image_size=(512, 512), patch_grid=(5, 5)):
        """
        Initialize the analyzer
        
        Args:
            image_size: tuple (width, height) of the image in pixels
            patch_grid: tuple (cols, rows) for patch grid division
        """
        self.image_size = image_size
        self.patch_grid = patch_grid
        self.patch_width = image_size[0] / patch_grid[0]
        self.patch_height = image_size[1] / patch_grid[1]
        
    def load_all_bbox_files(self, data_folder):
        """
        Load all *_bbox.json files from the data folder
        
        Args:
            data_folder: path to folder containing *_bbox.json files
            
        Returns:
            Combined dictionary with all bbox data
        """
        bbox_files = glob.glob(os.path.join(data_folder, "*_bbox.json"))
        print(f"Found {len(bbox_files)} bbox files")
        
        all_bbox_data = {}
        
        for file_path in bbox_files:
            try:
                with open(file_path, 'r') as f:
                    file_data = json.load(f)
                    all_bbox_data.update(file_data)
                    print(f"Loaded {len(file_data)} entries from {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        print(f"Total loaded entries: {len(all_bbox_data)}")
        return all_bbox_data
    
    def load_bbox_data(self, json_file_path):
        """Load bounding box data from single JSON file"""
        with open(json_file_path, 'r') as f:
            return json.load(f)
    
    def create_patch_polygons(self):
        """Create 25 patch polygons for the 5x5 grid"""
        patches = []
        patch_info = []
        
        for row in range(self.patch_grid[1]):  # 5 rows
            for col in range(self.patch_grid[0]):  # 5 cols
                # Calculate patch boundaries in normalized coordinates (0-1)
                x_min = col / self.patch_grid[0]
                y_min = row / self.patch_grid[1]
                x_max = (col + 1) / self.patch_grid[0]
                y_max = (row + 1) / self.patch_grid[1]
                
                # Create polygon
                patch_polygon = box(x_min, y_min, x_max, y_max)
                patches.append(patch_polygon)
                
                # Store patch info
                patch_info.append({
                    'patch_id': row * self.patch_grid[0] + col,
                    'row': row,
                    'col': col,
                    'x_min': x_min,
                    'y_min': y_min,
                    'x_max': x_max,
                    'y_max': y_max
                })
        
        # Create GeoDataFrame
        patches_gdf = gpd.GeoDataFrame(patch_info, geometry=patches)
        return patches_gdf
    
    def bbox_to_polygon(self, bbox):
        """
        Convert bounding box [x, y, width, height] to polygon
        
        Args:
            bbox: list [x, y, width, height] in normalized coordinates (0-1)
        """
        x, y, w, h = bbox
        return box(x, y, x + w, y + h)
    
    def analyze_single_image(self, bbox_data, image_path):
        """
        Analyze overlap for a single image
        
        Args:
            bbox_data: dictionary with image paths as keys and bbox lists as values
            image_path: specific image path to analyze
            
        Returns:
            GeoDataFrame with patch overlap counts
        """
        if image_path not in bbox_data:
            raise ValueError(f"Image path {image_path} not found in bbox data")
        
        # Create patch polygons
        patches_gdf = self.create_patch_polygons()
        
        # Get bounding box for this image
        bbox = bbox_data[image_path]
        rect_polygon = self.bbox_to_polygon(bbox)
        
        # Count overlaps
        overlap_counts = []
        for idx, patch in patches_gdf.iterrows():
            if patch.geometry.intersects(rect_polygon):
                overlap_counts.append(1)
            else:
                overlap_counts.append(0)
        
        patches_gdf['overlap_count'] = overlap_counts
        patches_gdf['image_path'] = image_path
        
        return patches_gdf
    
    def analyze_multiple_images(self, bbox_data, image_paths=None):
        """
        Analyze overlap for multiple images
        
        Args:
            bbox_data: dictionary with image paths as keys and bbox lists as values
            image_paths: list of image paths to analyze (if None, analyze all)
            
        Returns:
            GeoDataFrame with cumulative patch overlap counts
        """
        if image_paths is None:
            image_paths = list(bbox_data.keys())
        
        # Create patch polygons
        patches_gdf = self.create_patch_polygons()
        
        # Initialize overlap counts
        cumulative_counts = np.zeros(len(patches_gdf))
        
        # Process each image
        for image_path in image_paths:
            if image_path not in bbox_data:
                print(f"Warning: {image_path} not found in bbox data")
                continue
                
            bbox = bbox_data[image_path]
            rect_polygon = self.bbox_to_polygon(bbox)
            
            # Count overlaps for this image
            for idx, patch in patches_gdf.iterrows():
                if patch.geometry.intersects(rect_polygon):
                    cumulative_counts[idx] += 1
        
        patches_gdf['overlap_count'] = cumulative_counts
        
        # Sort by overlap count (descending)
        patches_gdf = patches_gdf.sort_values('overlap_count', ascending=False).reset_index(drop=True)
        
        return patches_gdf
    
    def visualize_patch_counts(self, patches_gdf, title="Patch Overlap Counts"):
        """Visualize patch overlap counts as a heatmap"""
        # Create a 5x5 matrix for visualization
        count_matrix = np.zeros((self.patch_grid[1], self.patch_grid[0]))
        
        for idx, patch in patches_gdf.iterrows():
            row, col = patch['row'], patch['col']
            count_matrix[row, col] = patch['overlap_count']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(count_matrix, annot=True, fmt='.0f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Overlap Count'})
        plt.title(title)
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.tight_layout()
        plt.show()
    
    def save_results(self, patches_gdf, output_file):
        """Save results to CSV file"""
        # Convert to regular DataFrame for saving (remove geometry column)
        result_df = patches_gdf.drop(columns='geometry').copy()
        result_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
    def get_class_analysis(self, bbox_data):
        """
        Analyze overlap patterns by class (assuming class info is in file path)
        
        Returns:
            Dictionary with class-wise statistics
        """
        class_stats = defaultdict(list)
        
        # Extract class from file paths and group
        for image_path, bbox in bbox_data.items():
            # Extract class from path (assuming format like /path/to/class/image.jpg)
            try:
                class_name = image_path.split('/')[-2]  # Get parent directory name
                class_stats[class_name].append(bbox)
            except:
                class_stats['unknown'].append(bbox)
        
        return dict(class_stats)

    def get_patch_statistics(self, patches_gdf):
        """Get statistics about patch overlaps"""
        stats = {
            'total_patches': len(patches_gdf),
            'patches_with_overlap': len(patches_gdf[patches_gdf['overlap_count'] > 0]),
            'max_overlap_count': patches_gdf['overlap_count'].max(),
            'mean_overlap_count': patches_gdf['overlap_count'].mean(),
            'patch_utilization': len(patches_gdf[patches_gdf['overlap_count'] > 0]) / len(patches_gdf) * 100
        }
        return stats

# Main processing function for all files
def process_all_bbox_files(data_folder, output_folder=None):
    """
    Process all *_bbox.json files in the data folder
    
    Args:
        data_folder: folder containing *_bbox.json files
        output_folder: folder to save results (if None, uses data_folder)
    """
    if output_folder is None:
        output_folder = data_folder
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize analyzer
    analyzer = RectangleOverlapAnalyzer(image_size=(512, 512), patch_grid=(5, 5))
    
    # Load all bbox data
    print("Loading all bbox files...")
    all_bbox_data = analyzer.load_all_bbox_files(data_folder)
    
    if not all_bbox_data:
        print("No bbox data found!")
        return
    
    # Analyze all images together
    print("Analyzing overlaps for all images...")
    result = analyzer.analyze_multiple_images(all_bbox_data)
    
    # Save overall results
    overall_output = os.path.join(output_folder, "overall_patch_analysis.csv")
    analyzer.save_results(result, overall_output)
    
    # Get and print statistics
    stats = analyzer.get_patch_statistics(result)
    print("\nOverall Statistics:")
    print("-" * 40)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Print top patches
    print("\nTop 10 patches by overlap count:")
    print("-" * 40)
    top_patches = result[['patch_id', 'row', 'col', 'overlap_count']].head(10)
    print(top_patches.to_string(index=False))
    
    # Analyze by class
    print("\nAnalyzing by class...")
    class_data = analyzer.get_class_analysis(all_bbox_data)
    
    class_results = {}
    for class_name, class_bboxes in class_data.items():
        if len(class_bboxes) < 5:  # Skip classes with too few samples
            continue
            
        # Create temporary bbox data for this class
        temp_bbox_data = {}
        for i, bbox in enumerate(class_bboxes):
            temp_bbox_data[f"{class_name}_{i}"] = bbox
        
        # Analyze this class
        class_result = analyzer.analyze_multiple_images(temp_bbox_data)
        class_results[class_name] = class_result
        
        # Save class-specific results
        class_output = os.path.join(output_folder, f"{class_name}_patch_analysis.csv")
        analyzer.save_results(class_result, class_output)
        
        # Print class statistics
        class_stats = analyzer.get_patch_statistics(class_result)
        print(f"\nClass: {class_name} ({len(class_bboxes)} images)")
        print(f"  Patches with overlap: {class_stats['patches_with_overlap']}/25")
        print(f"  Max overlap count: {class_stats['max_overlap_count']}")
        print(f"  Mean overlap count: {class_stats['mean_overlap_count']:.2f}")
        print(f"  Patch utilization: {class_stats['patch_utilization']:.1f}%")
    
    # Create visualization for overall results
    try:
        print("\nCreating overall visualization...")
        analyzer.visualize_patch_counts(result, "Overall Patch Overlap Counts - All Classes")
        
        # Save the plot
        plt.savefig(os.path.join(output_folder, "overall_patch_heatmap.png"), 
                   dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {os.path.join(output_folder, 'overall_patch_heatmap.png')}")
        
    except Exception as e:
        print(f"Visualization error: {e}")
    
    return result, class_results
# Example usage
def main():
    # Example: Process all files in a data folder
    data_folder = "C:/Users/ensin/OneDrive/Documenten/Universiteit/Thesis/MasterThesis/ThesisFleur/Falcon/FALcon-main/PSOL/results/ImageNet_train_set/VGG16-448"  # Change this to your actual path
    output_folder = "C:/Users/ensin/OneDrive/Documenten/Universiteit/Thesis/MasterThesis/ThesisFleur/Falcon/geopandas"   # Optional: specify output folder
    
    # Process all bbox files
    overall_result, class_results = process_all_bbox_files(data_folder, output_folder)
    
    print("Processing complete!")
    
    # Example of accessing results
    print("\nExample: Top 5 most active patches overall:")
    print(overall_result[['patch_id', 'row', 'col', 'overlap_count']].head(5).to_string(index=False))

# Alternative: Direct usage example
def example_usage():
    """Example showing direct usage of the analyzer"""
    # Initialize analyzer
    analyzer = RectangleOverlapAnalyzer(image_size=(512, 512), patch_grid=(5, 5))
    
    # Load all bbox files from a folder
    data_folder = "/path/to/your/data/folder"
    all_bbox_data = analyzer.load_all_bbox_files(data_folder)
    
    # Analyze overlaps
    result = analyzer.analyze_multiple_images(all_bbox_data)
    
    # Print results
    print("Top 10 patches by overlap count:")
    print(result[['patch_id', 'row', 'col', 'overlap_count']].head(10))
    
    # Get statistics
    stats = analyzer.get_patch_statistics(result)
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    return result

if __name__ == "__main__":
    main()