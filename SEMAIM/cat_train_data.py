import torch

class SingleImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, transform=None):
        from PIL import Image
        self.img = Image.open(img_path).convert("RGB")
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.img)
        return self.img
