import os
import json
import random
from PIL import Image
from tqdm import tqdm
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class ImageCaptioner:
    def __init__(self, image_paths, save_path="caption_map.json", save_interval=50, max_blip_captions=20):
        self.image_paths = image_paths
        self.save_path = save_path
        self.save_interval = save_interval
        self.max_blip_captions = max_blip_captions

        # Load caption cache nếu có
        self.caption_map = {}
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                self.caption_map = json.load(f)

        # BLIP setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

        # Caption fallback
        self.adjectives = ["smiling", "serious", "young", "old", "beautiful", "handsome", "confident", "shy"]
        self.genders = ["man", "woman", "person"]
        self.extras = ["with glasses", "wearing a hat", "with earrings", "with long hair", "with short hair", "looking left", "looking forward"]

    def blip_caption(self, path):
        try:
            image = Image.open(path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_length=30)
            return self.processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Lỗi caption ảnh {path}: {e}")
            return None

    def random_caption(self):
        return f"a {random.choice(self.adjectives)} {random.choice(self.genders)} {random.choice(self.extras)}"

    def generate_captions(self):
        blip_count = 0
        for i, path in enumerate(tqdm(self.image_paths)):
            fname = os.path.basename(path)
            if fname in self.caption_map:
                continue

            if self.max_blip_captions is None or blip_count < self.max_blip_captions:
                caption = self.blip_caption(path)
                if caption:
                    self.caption_map[fname] = caption
                    blip_count += 1
                    continue

            self.caption_map[fname] = self.random_caption()

            if i % self.save_interval == 0:
                self._save_caption_map()

        self._save_caption_map()
        print(f"Đã tạo caption cho {len(self.caption_map)} ảnh (BLIP: {blip_count}, random: {len(self.caption_map) - blip_count})")

    def _save_caption_map(self):
        with open(self.save_path, "w") as f:
            json.dump(self.caption_map, f, indent=2)

    def show_ramdom_sample_captions(self, n=5):
      # Chọn ngẫu nhiên n ảnh từ caption_map
      sample = random.sample(list(self.caption_map.items()), min(n, len(self.caption_map)))

      # Tìm lại đường dẫn ảnh tương ứng
      sample_paths = []
      for fname, caption in sample:
          for path in self.image_paths:
              if os.path.basename(path) == fname:
                  sample_paths.append((path, caption))
                  break

      # Hiển thị ảnh + caption
      plt.figure(figsize=(16, 4))
      for i, (path, caption) in enumerate(sample_paths):
          img = mpimg.imread(path)
          plt.subplot(1, n, i + 1)
          plt.imshow(img)
          plt.axis('off')
          plt.title(caption, fontsize=10)

      plt.tight_layout()
      plt.show()

    def show_random_from_first(self, from_top_n=20):
      # Chọn 1 ảnh ngẫu nhiên từ from_top_n ảnh đầu tiên
      sample_path = random.choice(self.image_paths[:from_top_n])
      fname = os.path.basename(sample_path)
      caption = self.caption_map.get(fname, "No caption")

      # Load ảnh
      img = mpimg.imread(sample_path)

      # Hiển thị ảnh
      plt.figure(figsize=(6, 6))
      plt.imshow(img)
      plt.axis("off")
      plt.title(caption, fontsize=14, wrap=True)
      plt.tight_layout()
      plt.show()
