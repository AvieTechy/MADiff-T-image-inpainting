import os
import json
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel

class TextEmbedder:
    def __init__(self, model_name="t5-small", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name).encoder.to(self.device).eval()

    def embed_text(self, text):
        with torch.no_grad():
            tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=20).to(self.device)
            output = self.model(**tokens).last_hidden_state.mean(dim=1).cpu()  # [1, hidden_size]
            return output.squeeze(0)

    def compute_all(self, caption_map, output_path="text_embeddings.pt", save=True):
        embeddings = {}
        for fname, caption in tqdm(caption_map.items(), desc="Embedding captions"):
            try:
                embeddings[fname] = self.embed_text(caption)
            except Exception as e:
                print(f"⚠️ Lỗi khi embed caption [{fname}]: {e}")
        if save:
            torch.save(embeddings, output_path)
            print(f"💾 Đã lưu embedding vào {output_path}")
        return embeddings

    def load_embeddings(self, path="text_embeddings.pt"):
        if os.path.exists(path):
            return torch.load(path)
        else:
            raise FileNotFoundError(f"Không tìm thấy file: {path}")
