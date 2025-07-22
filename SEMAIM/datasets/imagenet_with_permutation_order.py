from torchvision.datasets import ImageFolder
import torch
import os

class ImageNetWithTokenOrder(ImageFolder):
    def __init__(self, root, transform=None, token_order_dir=None):
        super().__init__(root, transform=transform)
        self.token_order_dir = token_order_dir

    def __getitem__(self, index):
        image, label = super().__getitem__(index)

        # Construct token order path from index (or filename)
        image_path = self.samples[index][0]
        filename = os.path.basename(image_path)
        sample_id = os.path.splitext(filename)[0]  # e.g., 'n01440764_36'
        token_path = os.path.join(self.token_order_dir, f"{sample_id}_token_order.pt")

        if os.path.exists(token_path):
            token_order = torch.load(token_path)  # Tensor of shape [N]
        else:
            token_order = None  # Optionally, you can fill in a default permutation here

        return image, label, token_order
