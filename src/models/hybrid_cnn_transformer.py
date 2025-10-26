import torch
import torch.nn as nn
from .base_model import BaseModel

class SelfAttentionBlock(nn.Module):
    """Bloc d'attention simple (Transformer Encoder allégé)."""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        # x: [B, N, C]
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = h + x
        x = x + self.mlp(self.norm2(x))
        return x

class HybridCNNTransformer(BaseModel):
    """Modèle hybride CNN + Transformer."""
    def __init__(self, n_classes=29, p=0.3, num_heads=4):
        super().__init__(n_classes)

        # Extraction locale (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 48x48
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 24x24
        )

        # Projection en séquence pour attention
        self.proj = nn.Conv2d(128, 128, kernel_size=1)
        self.attn_block = SelfAttentionBlock(dim=128, num_heads=num_heads)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Classification
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        # CNN → [B,128,H,W]
        x = self.cnn(x)
        B, C, H, W = x.shape
        # Flatten spatialement → [B, N, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.attn_block(x)
        # Moyenne spatiale globale
        x = x.mean(dim=1)
        return self.fc(x)
