import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CelebAHQCaptionDataset(Dataset):
    def __init__(self, root_dir, caption_map, text_embeddings, transform=None, mask_generator=None):
        self.root_dir = root_dir
        self.caption_map = caption_map
        self.text_embeddings = text_embeddings
        self.transform = transform
        self.mask_generator = mask_generator  # dùng mask random
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        img_path = os.path.join(self.root_dir, fname)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)  # Tensor [3, H, W]

        # Mask từ MaskGenerator
        if self.mask_generator:
            mask_np = self.mask_generator.sample().squeeze()  # [H, W]
            mask = torch.from_numpy(mask_np).unsqueeze(0).float()  # [1, H, W]
        else:
            mask = torch.ones_like(image[:1, :, :])  # fallback toàn 1 (no mask)

        # Tạo ảnh bị che
        masked_image = image * mask

        # Lấy text embedding tương ứng
        text_embed = self.text_embeddings[fname]  # Tensor [dim]

        return masked_image, mask, image, text_embed, fname
