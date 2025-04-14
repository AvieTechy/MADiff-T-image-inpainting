from transformers import T5Tokenizer, T5EncoderModel
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_encoder = T5EncoderModel.from_pretrained("t5-small").eval().to(device)

@torch.no_grad()
def encode_prompt(prompt_list):
    tokens = tokenizer(prompt_list, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = t5_encoder(**tokens)
    embed = outputs.last_hidden_state.mean(dim=1)  # [B, 512]
    return embed

class MaskedContextEncoder(nn.Module):
    def __init__(self, image_size=512, patch_size=16, in_channels=4, embed_dim=256, depth=4, nhead=8):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, (image_size // patch_size) ** 2, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, image, mask):
        x = torch.cat([image, mask], dim=1)                          # [B, 4, 512, 512]
        x = self.patch_embed(x)                                      # [B, D, 32, 32]
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)                              # [B, 1024, D]

        patch_mask = F.interpolate(mask, size=(H, W), mode='nearest')\
                        .flatten(2).transpose(1, 2)                  # [B, 1024, 1]
        patch_mask = (patch_mask < 0.5).squeeze(-1)                  # [B, 1024] - boolean

        masked_tokens, masked_pos = [], []
        for b in range(B):
            idx = patch_mask[b]
            masked_tokens.append(x[b][idx])
            masked_pos.append(self.pos_embed[0][idx])

        max_len = max(len(t) for t in masked_tokens)
        padded_tokens = torch.zeros(B, max_len, D, device=x.device)
        padded_pos = torch.zeros_like(padded_tokens)
        for b in range(B):
            l = len(masked_tokens[b])
            padded_tokens[b, :l] = masked_tokens[b]
            padded_pos[b, :l] = masked_pos[b]

        out = self.transformer(padded_tokens + padded_pos)           # [B, max_len, D]
        return out

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim=256, text_dim=512, num_heads=8):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, image_tokens, text_embedding):
        assert image_tokens.ndim == 3, f"image_tokens phải là [B, N, D], nhận được: {image_tokens.shape}"
        assert text_embedding.ndim == 2, f"text_embedding phải là [B, D], nhận được: {text_embedding.shape}"

        text_tokens = self.text_proj(text_embedding).unsqueeze(1)  # [B, 1, D]
        attn_output, _ = self.cross_attn(image_tokens, text_tokens, text_tokens)
        x = self.norm1(image_tokens + attn_output)
        x = self.norm2(x + self.ffn(x))
        return x  # [B, N, D]

class LightUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(out_c)
            )
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dec2 = conv_block(128 + 64, 64)
        self.dec1 = nn.Conv2d(64, out_channels, 3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d2 = self.dec2(torch.cat([self.up(e2), e1], dim=1))
        return self.dec1(d2)

class DiffusionDecoder(nn.Module):
    def __init__(self, token_dim=256, text_dim=512, out_channels=3, img_size=512, patch_size=16):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.token_dim = token_dim
        self.patch_size = patch_size

        self.latent_tokens = nn.Parameter(torch.randn(1, self.num_patches, token_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, token_dim))
        self.text_proj = nn.Linear(text_dim, token_dim)

        decoder_layer = nn.TransformerDecoderLayer(d_model=token_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.to_latent = nn.Linear(token_dim, patch_size * patch_size * out_channels)
        self.unet = LightUNet()

    def forward(self, context_token, text_embed):
        B = context_token.size(0)
        latent = self.latent_tokens.expand(B, -1, -1) + self.pos_embed
        text = self.text_proj(text_embed).unsqueeze(1)

        needed_len = self.num_patches + 1
        pad_len = needed_len - context_token.size(1)
        if pad_len > 0:
            padding = torch.zeros(B, pad_len, self.token_dim, device=context_token.device)
            context_token = torch.cat([context_token, padding], dim=1)

        memory = torch.cat([context_token, text], dim=1)
        x = self.transformer(latent, memory)
        x = self.to_latent(x)  # [B, 1024, 768]

        x = x.view(B, self.grid_size, self.grid_size, 3, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, 3, self.grid_size * self.patch_size, self.grid_size * self.patch_size)
        return self.unet(x)

class MADiffTModel(nn.Module):
    def __init__(self, encoder, cross_attn, decoder):
        super().__init__()
        self.encoder = encoder
        self.cross_attn = cross_attn
        self.decoder = decoder

    def forward(self, masked_img, mask, text_embed):
        context_token = self.encoder(masked_img, mask)               # [B, N, D]
        assert context_token.ndim == 3, "context_token phải là 3D"

        fused_token = self.cross_attn(context_token, text_embed)     # [B, N, D]
        out = self.decoder(fused_token, text_embed)                  # [B, 3, 512, 512]
        return out
