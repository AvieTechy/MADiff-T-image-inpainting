from transformers import T5Tokenizer, T5EncoderModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedContextEncoder(nn.Module):
    """
    Mã hóa ngữ cảnh có mask: chỉ xử lý các patch bị che
    """
    def __init__(self, image_size=512, patch_size=16, in_channels=4, embed_dim=256, depth=4, nhead=8):
        super().__init__()
        # Chuyển ảnh+mask thành embedding các patch
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Positional embedding cho mỗi patch
        self.pos_embed = nn.Parameter(torch.randn(1, (image_size // patch_size) ** 2, embed_dim))
        # Transformer encoder xử lý các token đã mask
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, image, mask):
        # Kết hợp ảnh và mask làm input
        x = torch.cat([image, mask], dim=1)                   # [B, 4, H, W]
        x = self.patch_embed(x)                               # [B, D, H/ps, W/ps]
        B, D, H, W = x.shape
        # flatten và chuẩn bị thành [B, num_patches, D]
        x = x.flatten(2).transpose(1, 2)                      # [B, N, D]

        # Tạo boolean mask cho patches bị che
        patch_mask = F.interpolate(mask, size=(H, W), mode='nearest') \
                        .flatten(2).transpose(1, 2)         # [B, N, 1]
        patch_mask = (patch_mask < 0.5).squeeze(-1)           # True nếu patch cần xử lý

        # Lấy token và vị trí của patch bị mask
        masked_tokens, masked_pos = [], []
        for b in range(B):
            idx = patch_mask[b]
            masked_tokens.append(x[b][idx])                  # token input
            masked_pos.append(self.pos_embed[0][idx])        # positional tương ứng

        # Padding để batch có độ dài đồng nhất
        max_len = max(t.size(0) for t in masked_tokens)
        padded_tokens = torch.zeros(B, max_len, D, device=x.device)
        padded_pos    = torch.zeros_like(padded_tokens)
        for b in range(B):
            l = masked_tokens[b].size(0)
            padded_tokens[b, :l] = masked_tokens[b]
            padded_pos[b, :l]    = masked_pos[b]

        # Chạy transformer trên các token đã mask
        out = self.transformer(padded_tokens + padded_pos)   # [B, max_len, D]
        return out


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention giữa feature ảnh và embedding văn bản
    """
    def __init__(self, embed_dim=256, text_dim=512, num_heads=8):
        super().__init__()
        # Projection từ embedding text về cùng chiều với image tokens
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        # Feed-forward sau attention
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, image_tokens, text_embedding):
        # Kết quả cross-attention
        text_tokens = self.text_proj(text_embedding).unsqueeze(1)  # [B, 1, D]
        attn_output, _ = self.cross_attn(image_tokens, text_tokens, text_tokens)
        # Residual & normalization
        x = self.norm1(image_tokens + attn_output)
        x = self.norm2(x + self.ffn(x))
        return x                                            # [B, N, D]


class StrongUNet(nn.Module):
    """
    UNet mạnh mẽ để tái tạo ảnh từ latent map
    """
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
        # Các tầng encoder
        self.enc1 = conv_block(in_channels, base_channels)
        self.enc2 = conv_block(base_channels, base_channels * 2)
        self.enc3 = conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = conv_block(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)
        # Các tầng decoder (upsample + concat)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = conv_block(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = conv_block(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = conv_block(base_channels * 2 + base_channels, base_channels)
        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Forward qua encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        # Forward qua decoder với skip connections
        d3 = self.dec3(torch.cat([self.up(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return self.final(d1)


class DiffusionDecoder(nn.Module):
    """
    Diffusion Decoder: từ context token + text embed sinh latent rồi UNet hồi lại ảnh
    """
    def __init__(self, token_dim=256, text_dim=512, out_channels=3, img_size=512, patch_size=16):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.token_dim = token_dim
        self.patch_size = patch_size
        # Latent tokens và positional embedding cho decoder
        self.latent_tokens = nn.Parameter(torch.randn(1, self.num_patches, token_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, token_dim))
        self.text_proj = nn.Linear(text_dim, token_dim)
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=token_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)
        # Chuyển token sang patch pixels
        self.to_latent = nn.Linear(token_dim, patch_size * patch_size * out_channels)
        # UNet để tái tạo ảnh full
        self.unet = StrongUNet()

    def forward(self, context_token, text_embed):
        B = context_token.size(0)
        # Khởi tạo latent sequence và gán positional
        latent = self.latent_tokens.expand(B, -1, -1) + self.pos_embed
        text = self.text_proj(text_embed).unsqueeze(1)  # [B,1,D]
        # Padding nếu thiếu token context
        needed = self.num_patches + 1
        pad_len = needed - context_token.size(1)
        if pad_len > 0:
            padding = torch.zeros(B, pad_len, self.token_dim, device=context_token.device)
            context_token = torch.cat([context_token, padding], dim=1)
        # Kết hợp context + text cho memory
        memory = torch.cat([context_token, text], dim=1)
        # Transformer decode
        x = self.transformer(latent, memory)
        # Mã về latent patches rồi reshape thành ảnh
        x = self.to_latent(x)
        x = x.view(B, self.grid_size, self.grid_size, 3, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, 3, img_size, img_size)
        return self.unet(x)


class MADiffTModel(nn.Module):
    """
    Mô hình tổng hợp: encoder -> cross-attn -> decoder
    """
    def __init__(self, encoder, cross_attn, decoder):
        super().__init__()
        self.encoder = encoder
        self.cross_attn = cross_attn
        self.decoder = decoder

    def forward(self, masked_img, mask, text_embed):
        # Mã hóa ngữ cảnh
        context_token = self.encoder(masked_img, mask)  # [B, N, D]
        # Kết hợp ngữ cảnh ảnh với embedding text
        fused = self.cross_attn(context_token, text_embed)  # [B, N, D]
        # Sinh ảnh cuối
        out = self.decoder(fused, text_embed)  # [B, 3, H, W]
        return out
