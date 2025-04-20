import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class CelebAHQCaptionDataset(Dataset):
    def __init__(self, image_paths, caption_map, text_embeddings,
                 transform=None, mask_generator=None):
        """
        - image_paths: list các đường dẫn tuyệt đối đến ảnh
        - caption_map: dict {fname: caption}
        - text_embeddings: dict {fname: Tensor}
        - transform: torchvision transform
        - mask_generator: object sinh mask (sample() -> [H, W, 1])
        """
        self.image_paths = image_paths
        self.caption_map = caption_map
        self.text_embeddings = text_embeddings
        self.transform = transform
        self.mask_generator = mask_generator

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        fname = os.path.basename(img_path)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)  # Tensor [3, H, W]

        # Mask từ MaskGenerator
        if self.mask_generator:
            mask_np = self.mask_generator.sample().squeeze()  # [H, W]
            mask = torch.from_numpy(mask_np).unsqueeze(0).float()  # [1, H, W]
        else:
            mask = torch.ones_like(image[:1, :, :])  # fallback toàn 1

        # Ảnh bị che
        masked_image = image * mask

        # Text embedding
        text_embed = self.text_embeddings[fname]  # Tensor [dim]

        return masked_image, mask, image, text_embed, fname
