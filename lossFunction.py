import torch
import torch.nn.functional as F
from torchvision.models import vgg16

# Load VGG16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = vgg16(pretrained=True).features[:16].eval().to(device)
for param in vgg.parameters():
    param.requires_grad = False

# VGG normalization constants (ImageNet)
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

def normalize_for_vgg(img):
    """ Chuẩn hoá ảnh đầu vào trước khi đưa vào VGG """
    return (img - mean) / std

def total_inpainting_loss(output, target, mask, λ=0.8):
    """
    output: [B, 3, H, W] - ảnh đã được inpaint
    target: [B, 3, H, W] - ground truth
    mask:   [B, 1, H, W] - mask (1: giữ nguyên, 0: cần phục hồi)
    λ: trọng số giữa L1 và perceptual loss
    """

    # Mở rộng mask sang 3 kênh nếu cần
    if mask.shape[1] == 1:
        mask = mask.expand(-1, 3, -1, -1)

    # Vùng bị mask là vùng cần phục hồi
    masked_region = (mask < 0.5).float()

    # L1 loss chỉ tính trên vùng bị mask
    l1_loss = F.l1_loss(output * masked_region, target * masked_region)

    # Perceptual loss
    vgg_output = vgg(normalize_for_vgg(output * masked_region))
    vgg_target = vgg(normalize_for_vgg(target * masked_region))
    perceptual_loss = F.l1_loss(vgg_output, vgg_target)

    # Tổng hợp loss
    total_loss = λ * l1_loss + (1 - λ) * perceptual_loss
    return total_loss
