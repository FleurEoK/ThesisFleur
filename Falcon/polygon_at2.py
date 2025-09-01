import json
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

class GridPolygonAnalyzer:
    def __init__(self, image_size=(512, 512), grid_size=(5, 5)):
        """
        Initialize the grid analyzer
        
        Args:
            image_size: tuple (width, height) of the image in pixels
            grid_size: tuple (cols, rows) for grid division
        """
        self.image_size = image_size
        self.grid_size = grid_size
        self.grid_width = 1.0 / grid_size[0]  # Normalized grid cell width
        self.grid_height = 1.0 / grid_size[1]  # Normalized grid cell height
        
    def load_all_bbox_files(self, data_folder):
        """Load all *_bbox.json files from the data folder"""
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
    
    def create_grid_cells(self):
        """
        Create grid cell polygons
        
        Returns:
            List of shapely box polygons representing grid cells
            Dictionary mapping (row, col) to polygon for easy access
        """
        grid_cells = []
        cell_map = {}
        
        for row in range(self.grid_size[1]):
            for col in range(self.grid_size[0]):
                # Calculate cell boundaries in normalized coordinates (0-1)
                x_min = col * self.grid_width
                y_min = row * self.grid_height
                x_max = (col + 1) * self.grid_width
                y_max = (row + 1) * self.grid_height
                
                # Create grid cell polygon
                cell_polygon = box(x_min, y_min, x_max, y_max)
                grid_cells.append(cell_polygon)
                cell_map[(row, col)] = cell_polygon
                
        return grid_cells, cell_map
    
    def bbox_to_polygon(self, bbox):
        """
        Convert bounding box [x, y, width, height] to polygon
        
        Args:
            bbox: list [x, y, width, height] in normalized coordinates (0-1)
        """
        x, y, w, h = bbox
        return box(x, y, x + w, y + h)
    
    def create_image_polygon(self, bbox_list):
        """
        Create a unified polygon from all regions in an image
        
        Args:
            bbox_list: list of bounding boxes for a single image
                      Can be a single bbox [x, y, w, h] or list of bboxes
                      
        Returns:
            Single polygon representing the union of all bounding boxes
        """
        # Handle single bbox case
        if isinstance(bbox_list[0], (int, float)):
            return self.bbox_to_polygon(bbox_list)
        
        # Handle multiple bboxes case
        polygons = []
        for bbox in bbox_list:
            if len(bbox) == 4:  # Ensure valid bbox
                polygons.append(self.bbox_to_polygon(bbox))
        
        if not polygons:
            raise ValueError("No valid bounding boxes found")
        
        if len(polygons) == 1:
            return polygons[0]
        else:
            # Create union of all polygons for this image
            return unary_union(polygons)
    
    def process_single_image(self, bbox_data, image_path):
        """
        Process a single image and return its grid representation
        
        Args:
            bbox_data: dictionary with image paths as keys and bbox lists as values
            image_path: specific image path to process
            
        Returns:
            tuple: (2D numpy array representing overlap grid, image metadata)
        """
        if image_path not in bbox_data:
            raise ValueError(f"Image path {image_path} not found in bbox data")
        
        # Create unified polygon for this image
        bbox_list = bbox_data[image_path]
        image_polygon = self.create_image_polygon(bbox_list)
        
        # Create grid
        grid_cells, cell_map = self.create_grid_cells()
        
        # Initialize count grid
        count_grid = np.zeros(self.grid_size, dtype=int)
        
        # Count overlaps (for single image, this will be binary: 0 or 1)
        for row in range(self.grid_size[1]):
            for col in range(self.grid_size[0]):
                cell_polygon = cell_map[(row, col)]
                if image_polygon.intersects(cell_polygon):
                    count_grid[row, col] = 1
        
        # Create metadata
        num_regions = 1 if isinstance(bbox_list[0], (int, float)) else len(bbox_list)
        metadata = {
            'image_path': image_path,
            'num_regions': num_regions,
            'bbox_list': bbox_list,
            'total_active_cells': np.sum(count_grid),
            'coverage_percentage': (np.sum(count_grid) / count_grid.size) * 100
        }
        
        return count_grid, metadata
    
    def process_all_images_separately(self, bbox_data):
        """
        Process all images separately and return individual grids plus cumulative
        
        Args:
            bbox_data: dictionary with image paths as keys and bbox lists as values
            
        Returns:
            tuple: (individual_results_dict, cumulative_grid)
        """
        results = {}
        cumulative_grid = np.zeros(self.grid_size, dtype=int)
        
        print(f"Processing {len(bbox_data)} images separately...")
        
        for i, image_path in enumerate(bbox_data.keys()):
            try:
                grid, metadata = self.process_single_image(bbox_data, image_path)
                results[image_path] = {
                    'grid': grid,
                    'metadata': metadata
                }
                
                # Add to cumulative grid
                cumulative_grid += grid
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(bbox_data)} images")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        print(f"Successfully processed {len(results)} images")
        return results, cumulative_grid
    
    def save_individual_grids(self, results, output_folder, save_format='numpy'):
        """
        Save individual grids to files
        
        Args:
            results: dictionary from process_all_images_separately
            output_folder: folder to save results
            save_format: 'numpy', 'csv', or 'both'
        """
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"Saving {len(results)} individual grids to {output_folder}")
        
        for image_path, result in results.items():
            # Create safe filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            safe_name = "".join(c for c in base_name if c.isalnum() or c in ('-', '_'))
            
            grid = result['grid']
            
            if save_format in ['numpy', 'both']:
                numpy_file = os.path.join(output_folder, f"{safe_name}_grid.npy")
                np.save(numpy_file, grid)
            
            if save_format in ['csv', 'both']:
                csv_file = os.path.join(output_folder, f"{safe_name}_grid.csv")
                np.savetxt(csv_file, grid, delimiter=',', fmt='%d')
        
        print(f"Saved all grids as {save_format} files")
    
    def save_example_grids(self, results, output_folder, num_examples=5):
        """
        Save a few example grids with their visualizations
        
        Args:
            results: dictionary from process_all_images_separately
            output_folder: folder to save examples
            num_examples: number of examples to save
        """
        examples_folder = os.path.join(output_folder, "examples")
        os.makedirs(examples_folder, exist_ok=True)
        
        print(f"Saving {num_examples} example grids with visualizations...")
        
        example_items = list(results.items())[:num_examples]
        
        for i, (image_path, result) in enumerate(example_items):
            # Create safe filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            safe_name = "".join(c for c in base_name if c.isalnum() or c in ('-', '_'))
            
            grid = result['grid']
            metadata = result['metadata']
            
            # Save grid as numpy and CSV
            numpy_file = os.path.join(examples_folder, f"example_{i+1}_{safe_name}_grid.npy")
            csv_file = os.path.join(examples_folder, f"example_{i+1}_{safe_name}_grid.csv")
            
            np.save(numpy_file, grid)
            np.savetxt(csv_file, grid, delimiter=',', fmt='%d')
            
            # Create and save visualization
            plt.figure(figsize=(6, 5))
            sns.heatmap(grid, annot=True, fmt='d', cmap='Blues', 
                       cbar_kws={'label': 'Overlap'}, vmin=0, vmax=1)
            
            title = f"Example {i+1}: {base_name}\n{metadata['num_regions']} regions, {metadata['total_active_cells']} active cells"
            plt.title(title, fontsize=12)
            plt.xlabel('Column')
            plt.ylabel('Row')
            plt.tight_layout()
            
            # Save the plot
            plot_file = os.path.join(examples_folder, f"example_{i+1}_{safe_name}_grid.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()  # Close to avoid memory issues
            
            print(f"Saved example {i+1}: {safe_name}")
        
        print(f"All {len(example_items)} examples saved to {examples_folder}")
        return examples_folder
    
    def save_selected_grids_only(self, results, cumulative_grid, output_folder, num_examples=5):
        """
        Save only the cumulative grid and first 5 individual examples (6 total)
        Both as images and CSV files (12 files total)
        
        Args:
            results: dictionary from process_all_images_separately
            cumulative_grid: 2D numpy array with cumulative counts
            output_folder: folder to save files
            num_examples: number of individual examples to save
        """
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"Saving cumulative grid + {num_examples} individual examples (6 grids total)")
        
        # 1. Save cumulative grid
        cumulative_csv = os.path.join(output_folder, "cumulative_grid.csv")
        np.savetxt(cumulative_csv, cumulative_grid, delimiter=',', fmt='%d')
        
        # Create cumulative grid visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(cumulative_grid, annot=True, fmt='d', cmap='YlOrRd', 
                   cbar_kws={'label': 'Overlap Count'})
        plt.title(f"Cumulative Grid - All {len(results)} Images")
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.tight_layout()
        
        cumulative_png = os.path.join(output_folder, "cumulative_grid.png")
        plt.savefig(cumulative_png, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved cumulative grid: CSV and PNG")
        
        # 2. Save first 5 individual examples
        example_items = list(results.items())[:num_examples]
        
        for i, (image_path, result) in enumerate(example_items):
            # Create safe filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            safe_name = "".join(c for c in base_name if c.isalnum() or c in ('-', '_'))
            
            grid = result['grid']
            metadata = result['metadata']
            
            # Save individual grid as CSV
            csv_file = os.path.join(output_folder, f"individual_{i+1}_{safe_name}.csv")
            np.savetxt(csv_file, grid, delimiter=',', fmt='%d')
            
            # Create and save individual visualization
            plt.figure(figsize=(6, 5))
            sns.heatmap(grid, annot=True, fmt='d', cmap='Blues', 
                       cbar_kws={'label': 'Overlap'}, vmin=0, vmax=1)
            
            title = f"Individual {i+1}: {base_name}\n{metadata['num_regions']} regions, {metadata['total_active_cells']} active cells"
            plt.title(title, fontsize=10)
            plt.xlabel('Column')
            plt.ylabel('Row')
            plt.tight_layout()
            
            # Save the plot
            png_file = os.path.join(output_folder, f"individual_{i+1}_{safe_name}.png")
            plt.savefig(png_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved individual {i+1}: {safe_name} (CSV + PNG)")
        
        total_files = 2 + (num_examples * 2)  # cumulative (2) + individuals (2 each)
        print(f"Total files saved: {total_files} ({total_files//2} CSVs + {total_files//2} PNGs)")
        
        return output_folder
    
    def save_metadata_summary(self, results, output_file):
        """
        Save metadata summary for all images
        
        Args:
            results: dictionary from process_all_images_separately
            output_file: path to save CSV summary
        """
        summary_data = []
        
        for image_path, result in results.items():
            metadata = result['metadata']
            summary_data.append({
                'image_path': image_path,
                'image_name': os.path.basename(image_path),
                'num_regions': metadata['num_regions'],
                'total_active_cells': metadata['total_active_cells'],
                'coverage_percentage': metadata['coverage_percentage']
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(output_file, index=False)
        print(f"Metadata summary saved to {output_file}")
        
        return df
    
    def visualize_individual_grid(self, grid, title="Individual Image Grid"):
        """
        Visualize a single image grid
        
        Args:
            grid: 2D numpy array to visualize
            title: title for the plot
        """
        plt.figure(figsize=(6, 5))
        sns.heatmap(grid, annot=True, fmt='d', cmap='Blues', 
                   cbar_kws={'label': 'Overlap'}, vmin=0, vmax=1)
        plt.title(title)
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.tight_layout()
        plt.show()
    
    def visualize_multiple_examples(self, results, num_examples=6):
        """
        Visualize grids from multiple example images
        
        Args:
            results: dictionary from process_all_images_separately
            num_examples: number of examples to show
        """
        example_items = list(results.items())[:num_examples]
        
        rows = 2
        cols = 3
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (image_path, result) in enumerate(example_items):
            if i >= num_examples:
                break
                
            grid = result['grid']
            metadata = result['metadata']
            
            sns.heatmap(grid, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[i], cbar=False, vmin=0, vmax=1)
            
            title = f"{os.path.basename(image_path)}\n{metadata['num_regions']} regions, {metadata['total_active_cells']} cells"
            axes[i].set_title(title, fontsize=10)
            axes[i].set_xlabel('Column')
            axes[i].set_ylabel('Row')
        
        # Hide unused subplots
        for i in range(num_examples, len(axes)):
            axes[i].hide()
        
        plt.tight_layout()
        plt.show()
    
    def grid_to_image_array(self, grid, scale_factor=1):
        """
        Convert grid to image array suitable for image conversion
        
        Args:
            grid: 2D numpy array representing the grid
            scale_factor: factor to scale up the grid (for higher resolution)
            
        Returns:
            2D numpy array scaled appropriately for image conversion
        """
        if scale_factor == 1:
            return grid.astype(np.uint8)
        else:
            # Scale up the grid by repeating each cell
            scaled_grid = np.repeat(np.repeat(grid, scale_factor, axis=0), scale_factor, axis=1)
            return scaled_grid.astype(np.uint8)
    
    def get_overall_statistics(self, results):
        """
        Get statistics across all processed images
        
        Args:
            results: dictionary from process_all_images_separately
            
        Returns:
            Dictionary with overall statistics
        """
        total_images = len(results)
        total_regions = sum(r['metadata']['num_regions'] for r in results.values())
        active_cells = [r['metadata']['total_active_cells'] for r in results.values()]
        coverage_percentages = [r['metadata']['coverage_percentage'] for r in results.values()]
        
        stats = {
            'total_images': total_images,
            'total_regions': total_regions,
            'mean_regions_per_image': total_regions / total_images if total_images > 0 else 0,
            'mean_active_cells': np.mean(active_cells),
            'median_active_cells': np.median(active_cells),
            'mean_coverage_percentage': np.mean(coverage_percentages),
            'images_with_single_region': sum(1 for r in results.values() if r['metadata']['num_regions'] == 1),
            'images_with_multiple_regions': sum(1 for r in results.values() if r['metadata']['num_regions'] > 1)
        }
        
        return stats

def main():
    """Main function to process all images separately"""
    
    # Initialize analyzer
    analyzer = GridPolygonAnalyzer(image_size=(512, 512), grid_size=(5, 5))
    
    # Load bbox data
    data_folder = "C:/Users/ensin/OneDrive/Documenten/Universiteit/Thesis/MasterThesis/ThesisFleur/Falcon/FALcon-main/PSOL/results/ImageNet_train_set/VGG16-448"
    bbox_data = analyzer.load_all_bbox_files(data_folder)
    
    if not bbox_data:
        print("No data loaded!")
        return
    
    # Process all images separately
    results, cumulative_grid = analyzer.process_all_images_separately(bbox_data)
    
    if not results:
        print("No images processed successfully!")
        return
    
    # Show overall statistics
    stats = analyzer.get_overall_statistics(results)
    print("\nOverall Statistics:")
    print("-" * 40)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    print(f"\nCumulative Grid:")
    print(cumulative_grid)
    
    # Setup output folder
    output_folder = "C:/Users/ensin/OneDrive/Documenten/Universiteit/Thesis/MasterThesis/ThesisFleur/Falcon/individual_grids"
    
    # Save only the 6 selected grids (cumulative + 5 examples) as both CSV and PNG
    analyzer.save_selected_grids_only(results, cumulative_grid, output_folder, num_examples=5)
    
    # Show examples
    print(f"\nShowing visualizations...")
    analyzer.visualize_multiple_examples(results, num_examples=6)  # Show cumulative + 5 examples
    
    # Show individual examples
    first_image_path = list(results.keys())[0]
    first_result = results[first_image_path]
    print(f"\nFirst image details:")
    print(f"Path: {first_image_path}")
    print(f"Grid:\n{first_result['grid']}")
    print(f"Metadata: {first_result['metadata']}")
    
    # Example: Convert grid to image array
    image_array = analyzer.grid_to_image_array(first_result['grid'], scale_factor=10)
    print(f"Image array shape for conversion: {image_array.shape}")
    
    return results, cumulative_grid

def example_single_image_processing():
    """Example of processing just one image"""
    
    analyzer = GridPolygonAnalyzer(image_size=(512, 512), grid_size=(5, 5))
    
    # Example bbox data for one image
    example_bbox_data = {
        'example_image.jpg': [0.2, 0.3, 0.4, 0.3]  # Single region
    }
    
    # Process single image
    grid, metadata = analyzer.process_single_image(example_bbox_data, 'example_image.jpg')
    
    print("Single Image Example:")
    print(f"Grid:\n{grid}")
    print(f"Metadata: {metadata}")
    
    # Visualize
    analyzer.visualize_individual_grid(grid, "Example Single Image")
    
    return grid, metadata

if __name__ == "__main__":
    # Run main processing
    results, cumulative_grid = main()
    
    # Run single image example
    print("\n" + "="*50)
    print("SINGLE IMAGE EXAMPLE")
    print("="*50)
    example_single_image_processing()