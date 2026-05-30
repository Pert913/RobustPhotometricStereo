"""
TransUNet, Lightweight, and SwinUNet architectures for Single-Image Input.

Architecture Flow (Single Image Configuration):
    - Input: 1 Color Image (3 channels RGB)
    - Part I:   CNN Encoder (Feature Extraction)
    - Part III: Transformer / Swin Bottleneck (Global Context)
    - Part II:  Robust Explicit Decoder
    - Head:     MultiTask Head predicting 4 channels (3 Normal + 1 Segmentation)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

# ==============================================================================
# TRANSFORMER COMPONENTS (TRANSUNET)
# ==============================================================================

class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, attention_dropout_rate=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.all_head_size = self.num_heads * self.head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.proj_dropout = nn.Dropout(attention_dropout_rate)

    def _transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        q = self._transpose_for_scores(self.query(hidden_states))
        k = self._transpose_for_scores(self.key(hidden_states))
        v = self._transpose_for_scores(self.value(hidden_states))

        attn_scores = torch.matmul(q, k.transpose(-1, -2))
        attn_scores = attn_scores / math.sqrt(self.head_size)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size(0), context.size(1), self.all_head_size)
        output = self.proj_dropout(self.out(context))
        return output

class Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(self.act(self.fc1(x)))))

class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_dim, dropout_rate=0.1, attention_dropout_rate=0.0):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, attention_dropout_rate)
        self.ffn = Mlp(hidden_size, mlp_dim, dropout_rate)

    def forward(self, x):
        x = x + self.attn(self.attention_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_dim, num_layers, dropout_rate=0.1, attention_dropout_rate=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_dim, dropout_rate, attention_dropout_rate)
            for _ in range(num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.encoder_norm(hidden_states)

class Embeddings(nn.Module):
    def __init__(self, in_channels, hidden_size, img_size=8, patch_size=1, dropout_rate=0.1):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        n_patches = (img_size // patch_size) ** 2
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.patch_embeddings(x)
        self._h, self._w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(-1, -2)
        
        seq_len = x.shape[1]
        if seq_len != self.position_embeddings.shape[1]:
            pos_embed = self._interpolate_pos_embed(self._h, self._w)
        else:
            pos_embed = self.position_embeddings
            
        return self.dropout(x + pos_embed)

    def _interpolate_pos_embed(self, target_h, target_w):
        pos = self.position_embeddings
        hidden = pos.shape[2]
        orig_len = pos.shape[1]
        orig_h = orig_w = int(math.sqrt(orig_len))
        pos = pos.reshape(1, orig_h, orig_w, hidden).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return pos.permute(0, 2, 3, 1).reshape(1, target_h * target_w, hidden)


# ==============================================================================
# CNN ENCODER & ROBUST DECODER
# ==============================================================================

class Conv2dReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv1 = Conv2dReLU(in_ch, out_ch)
        self.conv2 = Conv2dReLU(out_ch, out_ch)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.conv2(self.conv1(x)))

class SharedEncoder(nn.Module):
    def __init__(self, in_channels=3, channels=(64, 128, 256, 512), dropout=0.1):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, channels[0], dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = EncoderBlock(channels[0], channels[1], dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = EncoderBlock(channels[1], channels[2], dropout)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = EncoderBlock(channels[2], channels[3], dropout)
        self.pool4 = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        bottleneck = self.pool4(e4)
        return [e1, e2, e3, e4], bottleneck

class RobustDecoderBlock(nn.Module):
    """Robust Upsampler avoiding channel crashes"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class MultiTaskHead(nn.Module):
    """
    Decoupled Head for Multi-Task Learning.
    Separates the feature space for Normal Estimation and Mask Segmentation
    to prevent gradient domination in lightweight architectures.
    """
    def __init__(self, in_ch, out_ch=4):
        super().__init__()
        # Branch 1: Focus on learning 3D Normal (3 channels)
        self.normal_head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 3, kernel_size=1)
        )
        
        # Branch 2: Focus on learning background Segmentation (1 channel)
        self.mask_head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, 1, kernel_size=1)
        )

    def forward(self, x):
        normal_out = self.normal_head(x)
        mask_out = self.mask_head(x)
        # Concatenate 3 Normal channels and 1 Mask channel into a 4-channel Tensor
        return torch.cat([normal_out, mask_out], dim=1)


# ==============================================================================
# MODEL 1: TRANSUNET
# ==============================================================================

class TransUNetPS(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            from config import ModelConfig
            config = ModelConfig()

        enc_ch = getattr(config, 'encoder_channels', [64, 128, 256, 512])
        dec_ch = getattr(config, 'decoder_channels', [256, 128, 64, 16])
        hidden_size = getattr(config, 'hidden_size', 512)

        self.encoder = SharedEncoder(
            in_channels=getattr(config, 'in_channels', 3), 
            channels=enc_ch,
            dropout=getattr(config, 'dropout', 0.1),
        )

        self.embeddings = Embeddings(
            in_channels=enc_ch[-1], hidden_size=hidden_size,
            patch_size=1, dropout_rate=getattr(config, 'transformer_dropout', 0.1)
        )
        self.transformer_encoder = TransformerEncoder(
            hidden_size=hidden_size,
            num_heads=getattr(config, 'transformer_heads', 12),
            mlp_dim=getattr(config, 'transformer_ff_dim', 3072),
            num_layers=getattr(config, 'transformer_layers', 12),
            dropout_rate=getattr(config, 'transformer_dropout', 0.1),
            attention_dropout_rate=getattr(config, 'attention_dropout_rate', 0.0),
        )

        # TransUNet needs an adapter to match the 512 channels from the Transformer to the 256 decoder target
        self.adapter = Conv2dReLU(hidden_size, dec_ch[0])

        self.up4 = RobustDecoderBlock(in_ch=dec_ch[0], skip_ch=enc_ch[3], out_ch=dec_ch[1])
        self.up3 = RobustDecoderBlock(in_ch=dec_ch[1], skip_ch=enc_ch[2], out_ch=dec_ch[2])
        self.up2 = RobustDecoderBlock(in_ch=dec_ch[2], skip_ch=enc_ch[1], out_ch=dec_ch[3])
        self.up1 = RobustDecoderBlock(in_ch=dec_ch[3], skip_ch=enc_ch[0], out_ch=16)

        self.head = MultiTaskHead(16, 4)

    def forward(self, images):
        fused_skips, fused_bn = self.encoder(images)

        tokens = self.embeddings(fused_bn)
        h, w = self.embeddings._h, self.embeddings._w
        encoded = self.transformer_encoder(tokens)

        # Reshape 3D Tensor to 4D for the decoder
        B = encoded.shape[0]
        x = encoded.permute(0, 2, 1).contiguous().view(B, -1, h, w)
        
        x = self.adapter(x)
        x = self.up4(x, fused_skips[3])
        x = self.up3(x, fused_skips[2])
        x = self.up2(x, fused_skips[1])
        x = self.up1(x, fused_skips[0])

        return self.head(x)


# ==============================================================================
# MODEL 2: LIGHTWEIGHT UNET 
# ==============================================================================

class _LWConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x): return self.dropout(self.block(x))

class _LWDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _LWConvBlock(in_ch, out_ch, dropout)

    def forward(self, x): return self.conv(self.pool(x))

class LightweightUNetPS(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        in_ch = getattr(config, 'in_channels', 3)
        dropout = getattr(config, 'dropout', 0.1)
        enc_ch = [32, 64, 128, 256]
        bn_ch = 256

        self.enc1 = _LWConvBlock(in_ch, enc_ch[0], dropout)
        self.enc2 = _LWDownBlock(enc_ch[0], enc_ch[1], dropout)
        self.enc3 = _LWDownBlock(enc_ch[1], enc_ch[2], dropout)
        self.enc4 = _LWDownBlock(enc_ch[2], enc_ch[3], dropout)
        self.bottleneck = _LWDownBlock(enc_ch[3], bn_ch, dropout)

        self.patch_proj = nn.Conv2d(bn_ch, bn_ch, 1)
        self.pos_embed_row = nn.Embedding(64, bn_ch // 2)
        self.pos_embed_col = nn.Embedding(64, bn_ch // 2)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bn_ch, nhead=4, dim_feedforward=512, dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.transformer_norm = nn.LayerNorm(bn_ch)
        self.proj_back = nn.Conv2d(bn_ch, bn_ch, 1)

        self.up4 = RobustDecoderBlock(in_ch=bn_ch, skip_ch=enc_ch[3], out_ch=128)
        self.up3 = RobustDecoderBlock(in_ch=128, skip_ch=enc_ch[2], out_ch=64)
        self.up2 = RobustDecoderBlock(in_ch=64, skip_ch=enc_ch[1], out_ch=32)
        self.up1 = RobustDecoderBlock(in_ch=32, skip_ch=enc_ch[0], out_ch=32)

        self.head = MultiTaskHead(32, 4)

    def _add_pos_encoding(self, x):
        H, W = x.shape[2], x.shape[3]
        row_pos = self.pos_embed_row(torch.arange(H, device=x.device))
        col_pos = self.pos_embed_col(torch.arange(W, device=x.device))
        pos = torch.cat([
            row_pos.unsqueeze(1).expand(-1, W, -1),
            col_pos.unsqueeze(0).expand(H, -1, -1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0)
        return x + pos

    def forward(self, images):
        e1 = self.enc1(images)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        fused_bn = self.bottleneck(e4)

        x = self.patch_proj(fused_bn)
        residual = x
        x = self._add_pos_encoding(x)
        Bt, Ct, Ht, Wt = x.shape
        x = self.transformer_norm(self.transformer(x.flatten(2).permute(0, 2, 1)))
        x = self.proj_back(x.permute(0, 2, 1).view(Bt, Ct, Ht, Wt)) + residual

        x = self.up4(x, e4)
        x = self.up3(x, e3)
        x = self.up2(x, e2)
        x = self.up1(x, e1)
        return self.head(x)


# ==============================================================================
# MODEL 3: SWIN-UNET 
# ==============================================================================

class SwinUNetPS(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        enc_ch = [64, 128, 256, 512]
        
        self.encoder = SharedEncoder(
            in_channels=getattr(config, 'in_channels', 3),
            channels=enc_ch,
            dropout=getattr(config, 'dropout', 0.1),
        )

        bottleneck_dim = enc_ch[-1]

        self.patch_proj = nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=1)
        self.pos_row = nn.Embedding(64, bottleneck_dim // 2)
        self.pos_col = nn.Embedding(64, bottleneck_dim // 2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bottleneck_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.norm = nn.LayerNorm(bottleneck_dim)

        self.up4 = RobustDecoderBlock(in_ch=512, skip_ch=512, out_ch=256) 
        self.up3 = RobustDecoderBlock(in_ch=256, skip_ch=256, out_ch=128) 
        self.up2 = RobustDecoderBlock(in_ch=128, skip_ch=128, out_ch=64)  
        self.up1 = RobustDecoderBlock(in_ch=64,  skip_ch=64,  out_ch=32)  

        self.head = MultiTaskHead(in_ch=32, out_ch=4)

    def forward(self, images):
        fused_skips, fused_bn = self.encoder(images) 

        x = self.patch_proj(fused_bn)
        B, C, H, W = x.shape
        
        row = self.pos_row(torch.arange(H, device=x.device))
        col = self.pos_col(torch.arange(W, device=x.device))
        pos = torch.cat([
            row.unsqueeze(1).expand(-1, W, -1),
            col.unsqueeze(0).expand(H, -1, -1)
        ], dim=-1).permute(2, 0, 1).unsqueeze(0)
        
        x = x + pos
        residual = x
        
        x = x.flatten(2).permute(0, 2, 1) 
        x = self.norm(self.transformer(x))
        x = x.permute(0, 2, 1).view(B, C, H, W) 
        x = x + residual

        x = self.up4(x, fused_skips[3]) 
        x = self.up3(x, fused_skips[2]) 
        x = self.up2(x, fused_skips[1]) 
        x = self.up1(x, fused_skips[0]) 

        return self.head(x)

# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def get_model(config=None, model_type="transunet"):
    if model_type == "lightweight":
        return LightweightUNetPS(config)
    elif model_type == "swin":      
        return SwinUNetPS(config)
    else:
        return TransUNetPS(config)