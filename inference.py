def inference_first_image(model, image_dir, caption_map=None, mask_generator=None, device='cuda'):
    from glob import glob
    import os
    import torch
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from PIL import Image
    import numpy as np
    from transformers import T5Tokenizer, T5EncoderModel

    # 1. Lấy danh sách ảnh
    image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")) + glob(os.path.join(image_dir, "*.png")))
    if not image_paths:
        print(f"❌ Không tìm thấy ảnh trong: {image_dir}")
        return

    # 2. Lấy ảnh đầu tiên
    image_path = image_paths[2]
    fname = os.path.basename(image_path)
    print(f"🖼️ Ảnh đầu tiên: {fname}")

    # 3. Load ảnh
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, 512, 512]

    # 4. Caption
    caption = caption_map.get(fname, "a person") if caption_map else "a person"

    # 5. Tạo mask ngẫu nhiên nếu chưa có
    if mask_generator is None:
        from MaskGenerator import MaskGenerator
        mask_generator = MaskGenerator(height=512, width=512)

    mask_np = mask_generator.sample().transpose(2, 0, 1)
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).to(device).float()  # [1, 1, 512, 512]

    masked_img = img_tensor * mask_tensor  # [1, 3, 512, 512]

    # 6. Text embedding
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    text_encoder = T5EncoderModel.from_pretrained("t5-small").encoder.to(device).eval()
    tokens = tokenizer(caption, return_tensors="pt", truncation=True, max_length=20).to(device)
    with torch.no_grad():
        text_embed = text_encoder(**tokens).last_hidden_state.mean(dim=1)

    # 7. Inference
    model.eval()
    with torch.no_grad():
        output = model(masked_img, mask_tensor, text_embed)  # [1, 3, 512, 512]
        output = output.clamp(0, 1)

        # 🔁 Chỉ thay thế vùng bị che, giữ nguyên vùng còn lại từ ảnh gốc
        restored = img_tensor * mask_tensor + output * (1 - mask_tensor)

    # 8. Hiển thị ảnh
    def to_numpy(t): return t.squeeze(0).permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(to_numpy(img_tensor))
    plt.title("Ảnh gốc")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(to_numpy(masked_img))
    plt.title("Ảnh bị che")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(to_numpy(restored))
    plt.title(f"Phục hồi\n\"{caption}\"", fontsize=10)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
