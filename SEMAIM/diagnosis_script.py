import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse
from torchvision import transforms

# Import your model
from models import models_semaim as models_aim

def load_and_test_model(checkpoint_path, device='cpu'):
    """Load model and test its basic functionality"""
    print("🔍 Loading and diagnosing model...")
    
    # Load model
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

def test_model_outputs(model, test_image_path, device='cpu'):
    """Test what the model actually outputs"""
    print("🧪 Testing model outputs...")
    
    # Load test image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(test_image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    print(f"Input image shape: {image_tensor.shape}")
    print(f"Input image range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
    
    with torch.no_grad():
        # Test regular forward pass
        try:
            loss, permutation, loss_map = model(image_tensor, None, None)
            print(f"✅ Regular forward pass successful")
            print(f"   Loss: {loss.item():.4f}")
            print(f"   Permutation shape: {permutation.shape}")
            print(f"   Loss map shape: {loss_map.shape}")
            print(f"   Permutation range: [{permutation.min():.3f}, {permutation.max():.3f}]")
            print(f"   Loss map range: [{loss_map.min():.3f}, {loss_map.max():.3f}]")
        except Exception as e:
            print(f"❌ Regular forward pass failed: {e}")
            return
        
        # Test forward_aim
        try:
            pred, perm = model.forward_aim(image_tensor, None)
            print(f"✅ Forward_aim successful")
            print(f"   Pred shape: {pred.shape}")
            print(f"   Pred range: [{pred.min():.3f}, {pred.max():.3f}]")
            
            # Test if model has unpatchify
            if hasattr(model, 'unpatchify'):
                pred_patches = pred[:, 1:, :]  # Remove CLS token
                try:
                    reconstruction = model.unpatchify(pred_patches)
                    print(f"✅ Unpatchify successful")
                    print(f"   Reconstruction shape: {reconstruction.shape}")
                    print(f"   Reconstruction range: [{reconstruction.min():.3f}, {reconstruction.max():.3f}]")
                    
                    # Check if reconstruction looks reasonable
                    if reconstruction.min() < -10 or reconstruction.max() > 10:
                        print("⚠️  Reconstruction values seem extreme!")
                    
                    return reconstruction
                except Exception as e:
                    print(f"❌ Unpatchify failed: {e}")
            else:
                print("❌ Model has no unpatchify method")
                
        except Exception as e:
            print(f"❌ Forward_aim failed: {e}")

def check_training_data_quality(data_folder):
    """Check if training data looks reasonable"""
    print("📊 Checking training data quality...")
    
    import glob
    image_files = glob.glob(os.path.join(data_folder, "*.jpg"))
    
    if not image_files:
        print("❌ No training images found!")
        return
    
    print(f"Found {len(image_files)} training images")
    
    # Check a few images
    for i, img_path in enumerate(image_files[:3]):
        try:
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            
            print(f"Image {i+1}: {os.path.basename(img_path)}")
            print(f"  Size: {img.size}")
            print(f"  Array shape: {img_array.shape}")
            print(f"  Value range: [{img_array.min()}, {img_array.max()}]")
            print(f"  Mean RGB: {np.mean(img_array, axis=(0,1))}")
            
            # Check if image has reasonable content
            if np.std(img_array) < 10:
                print("⚠️  Image has very low variance - might be too uniform")
            
        except Exception as e:
            print(f"❌ Error reading {img_path}: {e}")

def create_simple_test_reconstruction(model, test_image_path, output_path):
    """Create a simple test reconstruction to see what the model produces"""
    print("🎨 Creating test reconstruction...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and process image
    image = Image.open(test_image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        # Get model outputs
        loss, permutation, loss_map = model(image_tensor, None, None)
        
        # Try to get reconstruction
        try:
            pred, perm = model.forward_aim(image_tensor, None)
            if hasattr(model, 'unpatchify'):
                pred_patches = pred[:, 1:, :]
                reconstruction = model.unpatchify(pred_patches)
                
                # Denormalize
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                recon_denorm = reconstruction * std + mean
                recon_denorm = torch.clamp(recon_denorm, 0, 1)
                
                # Save reconstruction
                from torchvision.transforms import ToPILImage
                recon_pil = ToPILImage()(recon_denorm[0])
                recon_pil.save(output_path)
                print(f"✅ Test reconstruction saved to {output_path}")
                
                # Create comparison
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(image)
                axes[0].set_title('Original')
                axes[0].axis('off')
                
                axes[1].imshow(recon_pil)
                axes[1].set_title('Reconstruction')
                axes[1].axis('off')
                
                comparison_path = output_path.replace('.png', '_comparison.png')
                plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"✅ Comparison saved to {comparison_path}")
                
        except Exception as e:
            print(f"❌ Could not create reconstruction: {e}")

def main():
    parser = argparse.ArgumentParser('Training Diagnosis')
    parser.add_argument('--checkpoint_path', default='./pretrain/aim_base/aim_base.exp_folder_permutations.temp.pth')
    parser.add_argument('--test_image', default='C:/Users/ensin/OneDrive/Documenten/Universiteit/Thesis/SemAIM/hutsel/preprocessed_cat.jpg', help='Path to test image')
    parser.add_argument('--training_data_folder', default='C:/Users/ensin/OneDrive/Documenten/Universiteit/Thesis/SemAIM/hutsel/permuted_images/', type=str, help='path to folder containing image permutations')
    parser.add_argument('--output_dir', default='./diagnosis_results')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🔬 Starting model diagnosis...")
    
    # Load model
    model = load_and_test_model(args.checkpoint_path)
    
    # Test model outputs
    test_model_outputs(model, args.test_image)
    
    # Check training data if provided
    if args.training_data_folder:
        check_training_data_quality(args.training_data_folder)
    
    # Create test reconstruction
    output_path = os.path.join(args.output_dir, 'test_reconstruction.png')
    create_simple_test_reconstruction(model, args.test_image, output_path)
    
    print("\n💡 RECOMMENDATIONS:")
    print("1. Check if loss was decreasing during training")
    print("2. Try training for more epochs (200-500)")
    print("3. Try a smaller learning rate (1e-5)")
    print("4. Check if the model architecture is appropriate for this task")
    print("5. Consider using a simpler baseline model first")

if __name__ == '__main__':
    main()