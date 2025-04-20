import os  # thao tác file hệ điều hành
import json  # đọc/ghi JSON
import random  # tạo caption ngẫu nhiên khi BLIP fail
from PIL import Image  # mở và xử lý ảnh
from tqdm import tqdm  # hiển thị progress bar
import torch  # kiểm tra device GPU/CPU
from transformers import BlipProcessor, BlipForConditionalGeneration  # BLIP captioning

import matplotlib.pyplot as plt  # vẽ ảnh
import matplotlib.image as mpimg  # đọc ảnh cho matplotlib

class ImageCaptioner:
    """
    Sinh caption cho ảnh bằng BLIP, với fallback random nếu BLIP thất bại.
    """
    def __init__(self, image_paths, save_path="caption_map.json", save_interval=50, max_blip_captions=20):
        # Danh sách đường dẫn ảnh cần caption
        self.image_paths = image_paths
        # File lưu trữ caption map
        self.save_path = save_path
        # Tần suất lưu file (mỗi `save_interval` ảnh)
        self.save_interval = save_interval
        # Số ảnh tối đa dùng BLIP, ảnh còn lại fallback random
        self.max_blip_captions = max_blip_captions

        # Load caption cache nếu đã có
        self.caption_map = {}
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                self.caption_map = json.load(f)

        # Thiết lập thiết bị (GPU nếu có)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Khởi tạo BLIP processor và model
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

        # Từ vựng phụ để tạo caption ngẫu nhiên khi cần
        self.adjectives = ["smiling", "serious", "young", "old", "beautiful", "handsome", "confident", "shy"]
        self.genders = ["man", "woman", "person"]
        self.extras = ["with glasses", "wearing a hat", "with earrings", "with long hair", "with short hair", "looking left", "looking forward"]

    def blip_caption(self, path):
        """
        Tạo caption cho 1 ảnh bằng BLIP.
        Trả về chuỗi caption hoặc None nếu có lỗi.
        """
        try:
            image = Image.open(path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_length=30)
            return self.processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Lỗi caption ảnh {path}: {e}")
            return None

    def random_caption(self):
        """
        Tạo caption ngẫu nhiên đơn giản khi BLIP fail hoặc vượt max_blip_captions.
        """
        return f"a {random.choice(self.adjectives)} {random.choice(self.genders)} {random.choice(self.extras)}"

    def generate_captions(self):
        """
        Duyệt qua tất cả ảnh, generate caption (BLIP hoặc random), và lưu map.
        """
        blip_count = 0
        for i, path in enumerate(tqdm(self.image_paths)):
            fname = os.path.basename(path)
            # Bỏ qua nếu đã có caption
            if fname in self.caption_map:
                continue

            # Dùng BLIP nếu vẫn còn quota
            if self.max_blip_captions is None or blip_count < self.max_blip_captions:
                caption = self.blip_caption(path)
                if caption:
                    self.caption_map[fname] = caption
                    blip_count += 1
                    continue

            # Fallback random
            self.caption_map[fname] = self.random_caption()

            # Lưu định kỳ
            if i % self.save_interval == 0:
                self._save_caption_map()

        # Lưu cuối cùng
        self._save_caption_map()
        print(f"Đã tạo caption cho {len(self.caption_map)} ảnh (BLIP: {blip_count}, random: {len(self.caption_map) - blip_count})")

    def _save_caption_map(self):
        """
        Lưu map {filename: caption} ra file JSON.
        """
        with open(self.save_path, "w") as f:
            json.dump(self.caption_map, f, indent=2)

    def show_ramdom_sample_captions(self, n=5):
        """
        Hiển thị ngẫu nhiên n ảnh kèm caption mẫu.
        """
        # Chọn n mẫu
        sample = random.sample(list(self.caption_map.items()), min(n, len(self.caption_map)))
        # Tìm đường dẫn tương ứng
        sample_paths = []
        for fname, caption in sample:
            for path in self.image_paths:
                if os.path.basename(path) == fname:
                    sample_paths.append((path, caption))
                    break

        # Vẽ ảnh + caption
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
        """
        Chọn ngẫu nhiên 1 ảnh trong top `from_top_n` và hiển thị kèm caption.
        """
        sample_path = random.choice(self.image_paths[:from_top_n])
        fname = os.path.basename(sample_path)
        caption = self.caption_map.get(fname, "No caption")

        img = mpimg.imread(sample_path)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(caption, fontsize=14, wrap=True)
        plt.tight_layout()
        plt.show()
