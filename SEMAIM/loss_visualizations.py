import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def find_log_file(output_dir):
    """Find the log file in the output directory"""
    print(f"Searching for log file in: {output_dir}")
    
    # List all files and find .txt files
    try:
        files = os.listdir(output_dir)
        log_files = [f for f in files if f.endswith('.log.txt')]
        
        if not log_files:
            print(" No .log.txt files found")
            print("Available files:")
            for file in sorted(files):
                print(f"  - {file}")
            return None
        
        # Use the first log file found
        log_file = log_files[0]
        log_path = os.path.join(output_dir, log_file)
        print(f" Found log file: {log_file}")
        return log_path
        
    except Exception as e:
        print(f"Error listing directory: {e}")
        return None

def read_loss_from_log_file(log_file_path):
    """Read loss data from the JSON log file"""
    epochs = []
    losses = []
    
    if not log_file_path or not os.path.exists(log_file_path):
        print(f"Log file not found: {log_file_path}")
        return epochs, losses
    
    print(f"Reading loss data from: {log_file_path}")
    
    with open(log_file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                line = line.strip()
                if not line:
                    continue
                    
                data = json.loads(line)
                
                # Look for loss and epoch data
                loss_value = None
                epoch_value = None
                
                # Check for loss keys
                for loss_key in ['train_loss', 'loss', 'avg_loss', 'epoch_loss']:
                    if loss_key in data:
                        loss_value = data[loss_key]
                        break
                
                # Check for epoch keys
                for epoch_key in ['epoch', 'step', 'iteration']:
                    if epoch_key in data:
                        epoch_value = data[epoch_key]
                        break
                
                if loss_value is not None and epoch_value is not None:
                    epochs.append(epoch_value)
                    losses.append(loss_value)
                elif line_num <= 5:  # Only show first few lines for debugging
                    print(f"Line {line_num}: Available keys: {list(data.keys())}")
                    
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: JSON decode error: {e}")
                continue
            except Exception as e:
                print(f"Line {line_num}: Error: {e}")
                continue
    
    print(f"Found {len(epochs)} epochs of loss data")
    if losses:
        print(f"Loss range: {min(losses):.4f} to {max(losses):.4f}")
    
    return epochs, losses

def plot_loss_curve(epochs, losses, output_path, title="Training Loss"):
    """Create and save loss curve plot"""
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # statistics
    if losses:
        min_loss = min(losses)
        min_epoch = epochs[losses.index(min_loss)]
        final_loss = losses[-1]
        
        plt.text(0.02, 0.98, f'Min Loss: {min_loss:.4f} (Epoch {min_epoch})\nFinal Loss: {final_loss:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss curve saved to: {output_path}")

def analyze_loss_trend(epochs, losses):
    """Analyze the loss trend and provide insights"""
    if not losses or len(losses) < 2:
        print("Not enough loss data for analysis")
        return
    
    print("\n=== Loss Analysis ===")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Min loss: {min(losses):.4f} (Epoch {epochs[losses.index(min(losses))]})") 
    print(f"Max loss: {max(losses):.4f} (Epoch {epochs[losses.index(max(losses))]})")
    
    # Check if loss is decreasing
    total_change = losses[-1] - losses[0]
    print(f"Total change: {total_change:.4f}")
    
    if total_change < -0.001:
        print("Loss is decreasing - model is learning!")
    elif abs(total_change) < 0.001:
        print(" Loss is relatively stable - model might not be learning much")
    else:
        print("Loss is increasing - there might be a training issue")
    
    # Check for convergence
    if len(losses) >= 10:
        recent_losses = losses[-10:]
        recent_std = np.std(recent_losses)
        if recent_std < 0.001:
            print(" Loss appears to have converged")
        else:
            print(f"Loss still changing (recent std: {recent_std:.4f})")
    
    # Provide recommendations based on loss value
    if losses[-1] > 1.0:
        print("  High final loss suggests:")
        print("   - Model might not be learning effectively")
        print("   - Try lower learning rate (1e-5 or 5e-5)")
        print("   - Train for more epochs")
        print("   - Check if data preprocessing is correct")
    elif losses[-1] < 0.1:
        print("Low final loss suggests good learning")
    else:
        print(f" Moderate final loss ({losses[-1]:.4f}) - could be better")

def main():
    parser = argparse.ArgumentParser('Simple Loss Visualization')
    parser.add_argument('--output_dir', default='C:/Users/ensin/OneDrive/Documenten/Universiteit/Thesis/SemAIM/pretrain/aim_base', 
                       help='Directory containing training outputs')
    parser.add_argument('--save_plot', default='./loss_curve.png', help='Where to save the plot')
    
    args = parser.parse_args()
    
    # Find and read the log file
    log_file_path = find_log_file(args.output_dir)
    
    if not log_file_path:
        print("No log file found!")
        return
    
    epochs, losses = read_loss_from_log_file(log_file_path)
    
    if not losses:
        print("No loss data found in log file!")
        return
    
    # Plot the loss curve
    plot_loss_curve(epochs, losses, args.save_plot, title="Training Loss")
    
    # Analyze the loss trend
    analyze_loss_trend(epochs, losses)
    
    print(f"Loss visualization complete!")
    print(f"Plot saved to: {args.save_plot}")

if __name__ == '__main__':
    main()