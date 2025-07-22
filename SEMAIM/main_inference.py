import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import your model
from models import models_semaim as models_aim
import util.misc as misc


def get_args_parser():
    parser = argparse.ArgumentParser('SemAIM Inference', add_help=False)
    
    # Model parameters
    parser.add_argument('--model', default='aim_base', type=str, help='Name of model')
    parser.add_argument('--input_size', default=224, type=int, help='Input image size')
    parser.add_argument('--query_depth', default=12, type=int, help='Decoder depth')
    parser.add_argument('--share_weight', action='store_true', help='Share weight between encoder and decoder')
    parser.add_argument('--prediction_head_type', default='MLP', type=str, help='Prediction head type')
    parser.add_argument('--gaussian_kernel_size', default=None, type=int, help='Gaussian blur kernel size')
    parser.add_argument('--gaussian_sigma', default=None, type=int, help='Gaussian blur sigma')
    parser.add_argument('--loss_type', default='L2', type=str, help='Loss type')
    parser.add_argument('--norm_pix_loss', action='store_true', help='Normalized pixel loss')
    parser.add_argument('--permutation_type', default='raster', type=str, help='Permutation type')
    parser.add_argument('--predict_feature', default='none', type=str, help='Feature prediction type')
    parser.add_argument('--attention_type', default='cls', type=str, help='Attention type')

    # Inference parameters
    parser.add_argument('--checkpoint_path', default='./pretrain/aim_base/aim_base.exp_folder_permutations.temp.pth', 
                        type=str, help='Path to trained checkpoint')
    parser.add_argument('--input_image', default=None, type=str, help='Path to input image for reconstruction')
    parser.add_argument('--permutation_folder', default='C:/Users/ensin/OneDrive/Documenten/Universiteit/Thesis/SemAIM/hutsel/permuted_images/', type=str, help='path to folder containing image permutations')
    parser.add_argument('--output_dir', default='./inference_results', type=str, help='Output directory for results')
    parser.add_argument('--device', default='cpu', type=str, help='Device to use for inference')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--save_reconstructions', action='store_true', help='Save reconstructed images')
    
    return parser


def load_model(args):
    """Load the trained model from checkpoint"""
    print(f"Loading model from {args.checkpoint_path}")
    
    # Create model
    out_dim = 512
    model = models_aim.__dict__[args.model](
        permutation_type=args.permutation_type,
        attention_type=args.attention_type,
        query_depth=args.query_depth,
        share_weight=args.share_weight,
        out_dim=out_dim,
        prediction_head_type=args.prediction_head_type,
        gaussian_kernel_size=args.gaussian_kernel_size,
        gaussian_sigma=args.gaussian_sigma,
        loss_type=args.loss_type,
        predict_feature=args.predict_feature,
        norm_pix_loss=args.norm_pix_loss
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    
    # Load state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from distributed training)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.to(args.device)
    model.eval()
    
    print(f"Model loaded successfully!")
    return model


def prepare_image(image_path, input_size=224):
    """Prepare image for inference"""
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor, image


def denormalize_image(tensor):
    """Denormalize image tensor for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    denormalized = tensor * std + mean
    denormalized = torch.clamp(denormalized, 0, 1)
    
    return denormalized


def reconstruct_image(model, image_tensor, device):
    """Reconstruct image using the trained model"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # Forward pass - the model should output reconstruction
        # Based on your training, the model takes (samples, enc_tokens, attention)
        # For inference, we can pass None for enc_tokens and attention
        loss, permutation, loss_map = model(image_tensor, None, None)
        
        print("Model outputs:")
        print(f"  Loss: {loss}")
        print(f"  Permutation shape: {permutation.shape if hasattr(permutation, 'shape') else type(permutation)}")
        print(f"  Loss map shape: {loss_map.shape if hasattr(loss_map, 'shape') else type(loss_map)}")
        
        # The permutation tensor [1, 196] contains patch predictions
        # We need to reshape and convert back to image format
        # 196 patches = 14x14 patches for 224x224 image (16x16 patches each)
        
        try:
            # Try to get the actual reconstruction from the model
            # The model might have a method to convert patch predictions back to image
            if hasattr(model, 'unpatchify'):
                # Convert patch predictions back to image
                reconstruction = model.unpatchify(permutation)
            else:
                # Manual reconstruction: reshape patch tokens to image
                batch_size = permutation.shape[0]
                num_patches = int(permutation.shape[1] ** 0.5)  # sqrt(196) = 14
                patch_size = 16  # Standard patch size for ViT
                
                # Reshape to patch grid
                # This is a simplified approach - the actual reconstruction depends on the model architecture
                reconstruction = permutation.view(batch_size, num_patches, num_patches, -1)
                
                # If the model outputs RGB values per patch, we need to reshape appropriately
                print(f"Permutation reshaped: {reconstruction.shape}")
                
                # For now, let's use the loss_map which might be more interpretable
                reconstruction = loss_map.view(batch_size, num_patches, num_patches)
                reconstruction = reconstruction.unsqueeze(1).repeat(1, 3, 1, 1)  # Make it 3-channel
                
                # Interpolate to original image size
                reconstruction = torch.nn.functional.interpolate(
                    reconstruction, 
                    size=(224, 224), 
                    mode='bilinear', 
                    align_corners=False
                )
        
        except Exception as e:
            print(f"Error in reconstruction: {e}")
            # Fallback: create a visualization from the patch tokens
            reconstruction = None
    
    return reconstruction, permutation, loss_map, loss


def visualize_results(original_image, reconstruction, permutation, loss_map, output_path):
    """Create visualization of reconstruction results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Reconstruction
    if reconstruction is not None:
        if len(reconstruction.shape) == 4:  # Batch dimension
            recon_img = reconstruction[0]
        else:
            recon_img = reconstruction
        
        if recon_img.shape[0] == 3:  # Channel first
            recon_img = recon_img.permute(1, 2, 0)
        
        recon_img = torch.clamp(recon_img, 0, 1)
        axes[0, 1].imshow(recon_img.cpu().numpy())
        axes[0, 1].set_title('Reconstructed Image')
        axes[0, 1].axis('off')
    
    # Permutation visualization
    if permutation is not None:
        if len(permutation.shape) == 2:  # Likely patch tokens
            perm_vis = permutation[0].cpu().numpy() if permutation.shape[0] > 1 else permutation.cpu().numpy()
            axes[1, 0].imshow(perm_vis.reshape(14, 14), cmap='viridis')  # 14x14 patches for 224x224 image
            axes[1, 0].set_title('Permutation Pattern')
            axes[1, 0].axis('off')
    
    # Loss map visualization
    if loss_map is not None:
        if len(loss_map.shape) == 2:
            loss_vis = loss_map[0].cpu().numpy() if loss_map.shape[0] > 1 else loss_map.cpu().numpy()
            axes[1, 1].imshow(loss_vis.reshape(14, 14), cmap='hot')
            axes[1, 1].set_title('Loss Map')
            axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    args = get_args_parser().parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args)
    
    # Determine input images
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
    
    # Process each image
    for i, image_path in enumerate(image_paths):
        print(f"\nProcessing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        # Prepare image
        image_tensor, original_image = prepare_image(image_path, args.input_size)
        
        # Reconstruct image
        reconstruction, permutation, loss_map, loss = reconstruct_image(model, image_tensor, device)
        
        # Save results
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        if args.save_reconstructions and reconstruction is not None:
            # Save reconstruction as image
            recon_denorm = denormalize_image(reconstruction)
            recon_pil = transforms.ToPILImage()(recon_denorm[0])
            recon_pil.save(os.path.join(args.output_dir, f"{base_name}_reconstruction.png"))
        
        if args.visualize:
            # Create visualization
            viz_path = os.path.join(args.output_dir, f"{base_name}_visualization.png")
            visualize_results(original_image, reconstruction, permutation, loss_map, viz_path)
        
        # Save reconstruction data as numpy arrays (more accessible than torch)
        import numpy as np
        save_data = {
            'reconstruction': reconstruction.cpu().numpy() if reconstruction is not None else None,
            'permutation': permutation.cpu().numpy(),
            'loss_map': loss_map.cpu().numpy(),
            'original_path': image_path,
            'loss_value': loss.item() if hasattr(loss, 'item') else float(loss)
        }
        
        # Save as numpy file
        np.save(os.path.join(args.output_dir, f"{base_name}_results.npy"), save_data)
        
        # Also save permutation as readable text
        with open(os.path.join(args.output_dir, f"{base_name}_permutation.txt"), 'w') as f:
            f.write(f"Image: {image_path}\n")
            f.write(f"Loss: {save_data['loss_value']}\n")
            f.write(f"Permutation shape: {permutation.shape}\n")
            f.write(f"Loss map shape: {loss_map.shape}\n")
            f.write(f"Permutation values:\n")
            perm_2d = permutation.cpu().numpy().reshape(14, 14)
            for row in perm_2d:
                f.write(" ".join([f"{val:.4f}" for val in row]) + "\n")
        
        print(f"Results saved for {base_name}")
        print(f"  Loss: {save_data['loss_value']:.4f}")
        print(f"  Reconstruction: {'Yes' if reconstruction is not None else 'No'}")
        
    
    print(f"\nInference completed! Results saved in {args.output_dir}")


if __name__ == '__main__':
    main()