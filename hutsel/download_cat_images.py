import os
import requests
import time
from pathlib import Path

def download_cat_images(num_images=16, folder_name="cat_images"):
    """
    Downloads random cat images from cataas.com and saves them to a folder
    
    Args:
        num_images (int): Number of cat images to download (default: 16)
        folder_name (str): Name of the folder to create (default: "cat_images")
    """
    
    # Create the folder if it doesn't exist
    folder_path = Path(folder_name)
    folder_path.mkdir(exist_ok=True)
    
    print(f"Downloading {num_images} cat images to '{folder_name}' folder...")
    
    successful_downloads = 0
    
    for i in range(num_images):
        try:
            # Get a random cat image from cataas.com
            response = requests.get("https://cataas.com/cat", timeout=10)
            
            if response.status_code == 200:
                # Save the image
                filename = f"cat_{i+1:02d}.jpg"
                filepath = folder_path / filename
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                successful_downloads += 1
                print(f"✓ Downloaded: {filename}")
                
            else:
                print(f"✗ Failed to download image {i+1}: HTTP {response.status_code}")
                
        except requests.RequestException as e:
            print(f"✗ Error downloading image {i+1}: {e}")
        
        # Small delay to be respectful to the server
        time.sleep(0.5)
    
    print(f"\nDownload complete! {successful_downloads}/{num_images} images downloaded successfully.")
    print(f"Images saved in: {folder_path.absolute()}")

if __name__ == "__main__":
    # You can customize these values
    NUM_IMAGES = 16
    FOLDER_NAME = "random_cat_images"
    
    download_cat_images(NUM_IMAGES, FOLDER_NAME)