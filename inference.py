import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, to_pil_image
from transformers import T5Tokenizer, T5EncoderModel
import cv2

def postprocess_restore_with_blur(output_tensor, mask_tensor, sharpen=True):
    """
    Hậu xử lý ảnh inpaint bằng cách làm mịn và làm sắc nét vùng được khôi phục.
    """
    output_np = to_pil_image(output_tensor.squeeze(0).cpu()).convert("RGB")
    mask_np = mask_tensor.squeeze().cpu().numpy()  # (H, W)

    img_cv = cv2.cvtColor(np.array(output_np), cv2.COLOR_RGB2BGR)
    mask_cv = (mask_np < 0.5).astype(np.uint8) * 255  # 1 ở vùng bị che

    # Làm mịn vùng mask
    blurred = cv2.GaussianBlur(img_cv, (5, 5), sigmaX=1.0)

    # Tùy chọn sharpen nhẹ
    if sharpen:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        blurred = cv2.filter2D(blurred, -1, kernel)

    # Trộn vùng mask bị che với ảnh đã blur
    alpha = np.repeat(mask_cv[..., None] / 255.0, 3, axis=2)
    result = blurred * alpha + img_cv * (1 - alpha)
    result = result.astype(np.uint8)

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result_tensor = to_tensor(Image.fromarray(result_rgb)).unsqueeze(0).to(mask_tensor.device)
    return result_tensor

def inference_single_image(img: Image.Image, mask: np.ndarray, caption: str, model, device="cuda"):
    """
    Inference 1 ảnh với mô hình MADiff-T
    """
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)

    # Resize và chuẩn hóa mask
    mask_resized = cv2.resize(mask.astype(np.float32), (512, 512))
    mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).unsqueeze(0).to(device).float()

    masked_img = img_tensor * mask_tensor

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    text_encoder = T5EncoderModel.from_pretrained("t5-small").encoder.to(device).eval()

    tokens = tokenizer(caption, return_tensors="pt", truncation=True, max_length=20).to(device)
    with torch.no_grad():
        text_embed = text_encoder(**tokens).last_hidden_state.mean(dim=1)

    with torch.no_grad():
        output = model(masked_img, mask_tensor, text_embed)
        output = output.clamp(0, 1)

        # Hậu xử lý nâng cao
        output = postprocess_restore_with_blur(output, mask_tensor)

        restored = img_tensor * mask_tensor + output * (1 - mask_tensor)

    # Hiển thị kết quả
    def to_numpy(t): return t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(to_numpy(img_tensor))
    plt.axis("off")
    plt.title("Ảnh gốc")

    plt.subplot(1, 3, 2)
    plt.imshow(to_numpy(masked_img))
    plt.axis("off")
    plt.title("Ảnh bị che")

    plt.subplot(1, 3, 3)
    plt.imshow(to_numpy(restored))
    plt.axis("off")
    plt.title(f"Phục hồi\n\"{caption}\"", fontsize=10)

    plt.tight_layout()
    plt.show()

    return restored
