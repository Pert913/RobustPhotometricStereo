"""
Configuration for TransUNetPS.

Updated to support Synthetic 300GB Dataset (8-8-8 LED Single Image Input) 
and backward compatibility with DiLiGenT Sparse Sampling.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """Dataset and data loading configuration."""
    data_root: str = "./data/training"
    
    # Old DiLiGenT defaults (Giữ nguyên để tương thích ngược)
    objects: List[str] = field(default_factory=lambda: [
        "ballPNG", "bearPNG", "buddhaPNG", "catPNG", "cowPNG",
        "gobletPNG", "harvestPNG", "pot1PNG", "pot2PNG", "readingPNG",
    ])
    num_lights: int = 96
    original_h: int = 512
    original_w: int = 612

    # Patch extraction
    patch_size: int = 128
    patches_per_epoch: int = 2000

    # --- CẬP NHẬT QUAN TRỌNG CHO SYNTHETIC 8-8-8 ---
    # Vì giờ ta đã ghép 3 hướng sáng vào 3 kênh R-G-B của 1 ảnh duy nhất,
    # đầu vào cho mô hình luôn luôn là 1 tấm ảnh.
    num_input_images: int = 1  
    min_input_images: int = 1   

    # Dataloader
    # Tăng từ 4 lên 16 vì 1 ảnh chiếm rất ít RAM so với nạp 12 ảnh như trước
    batch_size: int = 16 
    num_workers: int = 4  # Nếu chạy trên Windows bị lỗi đa luồng, hãy sửa lại thành 0


@dataclass
class ModelConfig:
    """
    TransUNetPS model configuration.
    """
    # Khóa cứng 3 kênh (Vì ảnh RGB 8-8-8 luôn có 3 kênh)
    in_channels: int = 3  
    
    encoder_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    decoder_channels: List[int] = field(default_factory=lambda: [512, 256, 128, 64, 16])
    out_channels: int = 3  # normal map (x, y, z)

    # Transformer encoder (Lightweight - ~5M params)
    hidden_size: int = 256
    transformer_heads: int = 4
    transformer_ff_dim: int = 512
    transformer_layers: int = 2
    transformer_dropout: float = 0.1
    attention_dropout_rate: float = 0.0

    # Decoder skip connections
    n_skip: int = 4

    # Regularization
    dropout: float = 0.1

    # Lightweight model defaults
    bottleneck_channels: int = 256

    # Mode selection
    mode: str = "normal"  # "normal" or "segmentation"
    num_classes: int = 9  # Synapse: 9 organ classes

    # Model selection
    model_type: str = "lightweight"  # "transunet" or "lightweight"


@dataclass
class TrainConfig:
    """Training configuration."""
    epochs: int = 100 # Giảm từ 150 xuống 100 vì dataset mới rất lớn, model học nhanh hơn
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    lr_power: float = 0.9
    optimizer: str = "adamw"

    # Loss weights (Photometric Stereo)
    angular_weight: float = 0.5
    cosine_weight: float = 0.3
    l1_weight: float = 0.2

    # Early stopping
    patience: int = 50 

    # Gradient clipping
    grad_clip: float = 1.0

    # Checkpointing
    save_dir: str = "./checkpoints"
    log_interval: int = 10


@dataclass
class Config:
    """Top-level configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = "auto"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"