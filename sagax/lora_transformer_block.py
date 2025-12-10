import torch
from torch import nn
from lora_mha import LoRAMultiHeadAttention  # your LoRA MHA

class LoRATransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, r=8, alpha=32, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # LayerNorm before MHA (Pre-LN)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mha = LoRAMultiHeadAttention(embed_dim, num_heads, r=r, alpha=alpha)

        # LayerNorm before FFN
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # MHA block
        x_res = x
        x = self.ln1(x)
        x = self.mha(x)
        x = self.dropout(x)
        x = x + x_res  # Residual

        # FFN block
        x_res = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + x_res  # Residual

        return x


batch_size = 2
seq_len = 5
embed_dim = 64
num_heads = 4
ffn_dim = 128
r = 4
alpha = 8

x = torch.randn(batch_size, seq_len, embed_dim)

block = LoRATransformerBlock(embed_dim, num_heads, ffn_dim, r=r, alpha=alpha)

out = block(x)

print("Input shape:", x.shape)
print("Output shape:", out.shape)

# Trainable parameters
print("\nTrainable LoRA parameters:")
for name, p in block.named_parameters():
    if p.requires_grad:
        print(name, p.shape)
