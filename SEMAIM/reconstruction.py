import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse

def load_model_with_reconstruction(checkpoint_path, device='cpu'):
    """Load model and add reconstruction capability"""
    from models import models_semaim as models_aim
    
    # Model parameters (should match your training)
    model = models_aim.aim_base(
        permutation_type='raster',
        attention_type='cls',
        query_depth=12,
        share_weight=False,
        out_dim=512,
        prediction_head_type='MLP',
        gaussian_kernel_size=None,
        gaussian_sigma=None,
        loss_type='L2',
        predict_feature='none',
        norm_pix_loss=False
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    return model

def get_model_reconstruction(model, image_tensor, device):
    """Get reconstruction by accessing model's internal forward pass"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # Call the model's forward_aim method directly to get intermediate outputs
        try:
            pred, permutation = model.forward_aim(image_tensor, None)
            print(f"Forward aim outputs - pred: {pred.shape}, permutation: {permutation.shape}")
            
            # The 'pred' should be the reconstructed patches
            # pred shape: [1, 197, 768] - need to remove CLS token and convert to patches
            
            # Remove CLS token (first token)
            pred_patches = pred[:, 1:, :]  # Shape: [1, 196, 768]
            print(f"After removing CLS token - pred_patches: {pred_patches.shape}")
            
            # Try to convert it back to image format
            if hasattr(model, 'unpatchify'):
                reconstruction = model.unpatchify(pred_patches)
                print(f"Model unpatchify reconstruction: {reconstruction.shape}")
            else:
                # The model's prediction head should convert 768-dim features to patch pixels
                # Let's check if the model has a prediction head
                if hasattr(model, 'prediction_head'):
                    # Apply prediction head to convert features to pixel values
                    patch_pixels = model.prediction_head(pred_patches)  # Should output RGB values
                    print(f"After prediction head: {patch_pixels.shape}")
                    
                    # Now unpatchify to reconstruct the image
                    reconstruction = unpatchify_manual(patch_pixels, model.patch_embed.patch_size)
                    print(f"Manual unpatchify reconstruction: {reconstruction.shape}")
                else:
                    # Fallback: try to reshape the features directly
                    print("No prediction head found, trying direct reshape...")
                    # This might not work perfectly but let's try
                    reconstruction = pred_patches.view(1, 14, 14, 768)
                    reconstruction = reconstruction.permute(0, 3, 1, 2)  # [1, 768, 14, 14]
                    
                    # Interpolate to image size
                    reconstruction = torch.nn.functional.interpolate(
                        reconstruction, size=(224, 224), mode='bilinear', align_corners=False
                    )
                    # Take only first 3 channels as RGB
                    reconstruction = reconstruction[:, :3, :, :]
                    print(f"Direct reshape reconstruction: {reconstruction.shape}")
            
            return reconstruction, pred_patches, permutation
            
        except Exception as e:
            print(f"Error in forward_aim: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to regular forward pass
            loss, permutation, loss_map = model(image_tensor, None, None)
            return None, permutation, loss_map

def unpatchify_manual(x, patch_size=16):
    """
    Manually unpatchify patches back to image
    x: (N, L, patch_size**2 * 3) where L = 196 for 14x14 patches
    """
    p = patch_size
    h = w = int(x.shape[1]**.5)  # sqrt(196) = 14
    assert h * w == x.shape[1], f"Expected {h*w} patches, got {x.shape[1]}"
    
    # Check if we have the right number of channels
    expected_channels = p * p * 3  # 16 * 16 * 3 = 768
    if x.shape[2] != expected_channels:
        print(f"Warning: Expected {expected_channels} channels, got {x.shape[2]}")
        # If we have different dimensions, we might need to project to the right size
        # This depends on your model architecture
        if x.shape[2] == 768:  # Feature dimension
            # This might be feature vectors, not pixel values
            # We need to convert features to pixels somehow
            # For now, let's just take the first 768 values and reshape
            pass
    
    try:
        # Reshape to patches: [N, H, W, P, P, C]
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        # Rearrange to image format: [N, C, H*P, W*P]
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 3, h * p, w * p)
        return imgs
    except Exception as e:
        print(f"Error in unpatchify_manual: {e}")
        print(f"Input shape: {x.shape}")
        raise

def visualize_reconstruction_process(original_image, reconstruction, patches, permutation, output_path):
    """Create detailed visualization of the reconstruction process"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Reconstruction
    if reconstruction is not None:
        if len(reconstruction.shape) == 4:
            recon_img = reconstruction[0]
        else:
            recon_img = reconstruction
        
        if recon_img.shape[0] == 3:  # Channel first
            recon_img = recon_img.permute(1, 2, 0)
        
        # Denormalize if needed
        if recon_img.min() < 0:  # Probably normalized
            recon_img = (recon_img + 1) / 2  # Assuming [-1, 1] normalization
        
        recon_img = torch.clamp(recon_img, 0, 1)
        axes[0, 1].imshow(recon_img.cpu().numpy())
        axes[0, 1].set_title('Reconstructed Image')
        axes[0, 1].axis('off')
    else:
        axes[0, 1].text(0.5, 0.5, 'No reconstruction\navailable', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].axis('off')
    
    # Patch visualization - handle the [1, 196, 768] shape properly
    if patches is not None:
        try:
            # patches shape: [1, 196, 768]
            # We want to visualize the patch features as a 14x14 grid
            # Let's take the mean of the 768 feature dimensions for each patch
            patch_features = patches[0].cpu().numpy()  # Shape: [196, 768]
            patch_means = np.mean(patch_features, axis=1)  # Shape: [196]
            patch_grid = patch_means.reshape(14, 14)  # Shape: [14, 14]
            
            im = axes[0, 2].imshow(patch_grid, cmap='viridis')
            axes[0, 2].set_title('Patch Feature Means')
            plt.colorbar(im, ax=axes[0, 2])
        except Exception as e:
            print(f"Error visualizing patches: {e}")
            axes[0, 2].text(0.5, 0.5, f'Patch visualization\nerror: {e}', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].axis('off')
    
    # Permutation pattern
    if permutation is not None:
        try:
            perm_grid = permutation[0].cpu().numpy()
            if len(perm_grid.shape) == 1:
                perm_grid = perm_grid.reshape(14, 14)
            
            im = axes[1, 0].imshow(perm_grid, cmap='plasma')
            axes[1, 0].set_title('Permutation Pattern')
            plt.colorbar(im, ax=axes[1, 0])
        except Exception as e:
            print(f"Error visualizing permutation: {e}")
            axes[1, 0].text(0.5, 0.5, f'Permutation visualization\nerror: {e}', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].axis('off')
    
    # Difference map (if reconstruction available)
    if reconstruction is not None:
        try:
            # Create a simple difference between original and reconstruction
            from torchvision import transforms
            
            # Convert original PIL image to tensor
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            original_tensor = transform(original_image).unsqueeze(0)
            
            # Denormalize reconstruction for comparison
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            recon_denorm = reconstruction * std + mean
            recon_denorm = torch.clamp(recon_denorm, 0, 1)
            
            # Calculate difference
            diff = torch.abs(original_tensor - recon_denorm)
            diff_img = diff[0].permute(1, 2, 0).cpu().numpy()
            diff_gray = np.mean(diff_img, axis=2)  # Convert to grayscale
            
            im = axes[1, 1].imshow(diff_gray, cmap='hot')
            axes[1, 1].set_title('Reconstruction Difference')
            plt.colorbar(im, ax=axes[1, 1])
        except Exception as e:
            print(f"Error creating difference map: {e}")
            axes[1, 1].text(0.5, 0.5, 'Difference Map\n(calculation error)', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].axis('off')
    else:
        axes[1, 1].text(0.5, 0.5, 'No reconstruction\navailable', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')
    
    # Statistics
    stats_text = f"Patch shape: {patches.shape if patches is not None else 'N/A'}\n"
    stats_text += f"Permutation shape: {permutation.shape if permutation is not None else 'N/A'}\n"
    if reconstruction is not None:
        stats_text += f"Reconstruction shape: {reconstruction.shape}\n"
        stats_text += f"Reconstruction range: [{reconstruction.min():.3f}, {reconstruction.max():.3f}]"
    
    axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
                   fontsize=10, verticalalignment='top')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Reconstruction successful! Shape: {reconstruction.shape if reconstruction is not None else 'None'}")

def main():
    parser = argparse.ArgumentParser('Patch Reconstruction Analysis')
    parser.add_argument('--checkpoint_path', default='./pretrain/aim_base/aim_base.exp_folder_cats.temp.pth')
    parser.add_argument('--input_image', default=None, help='Path to single input image')
    parser.add_argument('--permutation_folder', default=None, help='path to folder containing image permutations')
    parser.add_argument('--output_dir', default='./cat/reconstruction_analysis')
    parser.add_argument('--device', default='cpu')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model_with_reconstruction(args.checkpoint_path, args.device)
    
    # Prepare image transform
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # FOLDER PROCESSING LOGIC - THIS IS THE PART YOU KEEP ASKING ABOUT!
    if args.input_image:
        image_paths = [args.input_image]
    elif args.permutation_folder:
        import glob
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(args.permutation_folder, ext)))
            image_paths.extend(glob.glob(os.path.join(args.permutation_folder, ext.upper())))
        image_paths = sorted(list(set(image_paths)))  # Remove duplicates and sort
    else:
        raise ValueError("Please specify either --input_image or --permutation_folder")
    
    print(f"Found {len(image_paths)} images to process")
    
    # Process each image in the folder
    for i, image_path in enumerate(image_paths):
        print(f"\nProcessing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        try:
            # Load and prepare image - use image_path from loop, not args.input_image
            original_image = Image.open(image_path).convert('RGB')
            image_tensor = transform(original_image).unsqueeze(0)
            
            # Get reconstruction
            reconstruction, patches, permutation = get_model_reconstruction(model, image_tensor, args.device)
            
            # Create visualization
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            viz_path = os.path.join(args.output_dir, f"{base_name}_detailed_reconstruction_cat.png")
            
            visualize_reconstruction_process(original_image, reconstruction, patches, permutation, viz_path)
            
            print(f"Detailed reconstruction analysis saved to {viz_path}")
            
            # Save raw data
            if reconstruction is not None:
                # Save reconstruction as image
                from torchvision.transforms import ToPILImage
                
                # Denormalize
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                recon_denorm = reconstruction * std + mean
                recon_denorm = torch.clamp(recon_denorm, 0, 1)
                
                recon_pil = ToPILImage()(recon_denorm[0])
                recon_pil.save(os.path.join(args.output_dir, f"{base_name}_reconstruction_cat.png"))
                print(f"Reconstruction image saved for {base_name}")
            else:
                print(f"No reconstruction could be generated for {base_name}")
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nAll processing completed! Results saved in {args.output_dir}")

if __name__ == '__main__':
    main()


