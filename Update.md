# Update
⸻

✅ 1. UNet mạnh hơn thay LightUNet

class StrongUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(in_channels, base_channels)
        self.enc2 = conv_block(base_channels, base_channels * 2)
        self.enc3 = conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = conv_block(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec3 = conv_block(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = conv_block(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = conv_block(base_channels * 2 + base_channels, base_channels)

        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))

        return self.final(d1)



⸻

✅ 2. Loss nâng cao chỉ tính trên vùng mask

def enhanced_inpainting_loss(output, target, mask, λ=0.8, vgg=None, normalize_for_vgg=None):
    if mask.shape[1] == 1:
        mask = mask.expand(-1, 3, -1, -1)

    focus_mask = 1 - mask  # chỉ tính loss ở vùng bị che

    l1 = F.l1_loss(output * focus_mask, target * focus_mask)

    if vgg is not None and normalize_for_vgg is not None:
        perceptual = F.l1_loss(
            vgg(normalize_for_vgg(output * focus_mask)),
            vgg(normalize_for_vgg(target * focus_mask))
        )
    else:
        perceptual = 0

    return λ * l1 + (1 - λ) * perceptual



⸻

✅ 3. Hậu xử lý làm mịn vùng bị che (Gaussian Blur)

import cv2
import numpy as np
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image

def postprocess_restore_with_blur(output_tensor, mask_tensor):
    """
    Làm mịn vùng inpainting bằng Gaussian blur trên vùng mask.
    """
    output_np = to_pil_image(output_tensor.squeeze(0).cpu()).convert("RGB")
    mask_np = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()

    img_cv = cv2.cvtColor(np.array(output_np), cv2.COLOR_RGB2BGR)
    mask_cv = (mask_np < 0.5).astype(np.uint8) * 255  # 1 ở vùng bị che

    blurred = cv2.GaussianBlur(img_cv, (11, 11), 0)
    result = np.where(mask_cv[..., None] == 255, blurred, img_cv)

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result_tensor = to_tensor(Image.fromarray(result_rgb)).unsqueeze(0)
    return result_tensor



⸻

👉 Gợi ý tích hợp:
	•	Thay LightUNet bằng StrongUNet() trong DiffusionDecoder
	•	Gọi postprocess_restore_with_blur() sau khi nhận được output từ model
	•	Thay total_inpainting_loss() bằng enhanced_inpainting_loss() trong vòng huấn luyện

⸻

Nếu bạn muốn mình tích hợp tất cả vào DiffusionDecoder hoặc file training chính, mình có thể viết gọn sẵn cho bạn! ￼
