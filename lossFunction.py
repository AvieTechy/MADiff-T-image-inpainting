import torch
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

# Load VGG16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval().to(device)
for param in vgg.parameters():
    param.requires_grad = False

# VGG normalization constants (ImageNet)
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

def normalize_for_vgg(img):
    return (img - mean) / std

def total_inpainting_loss(output, target, mask, λ=0.8):
    if mask.shape[1] == 1:
        mask = mask.expand(-1, 3, -1, -1)

    masked_region = (mask < 0.5).float()
    l1_loss = F.l1_loss(output * masked_region, target * masked_region)

    vgg_output = vgg(normalize_for_vgg(output * masked_region))
    vgg_target = vgg(normalize_for_vgg(target * masked_region))
    perceptual_loss = F.l1_loss(vgg_output, vgg_target)

    return λ * l1_loss + (1 - λ) * perceptual_loss
