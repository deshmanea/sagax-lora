import math
from torch import nn
from lora import LoRALinear


class LoRAMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, r=8, alpha=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # LoRA only on Q and V
        self.q_proj = LoRALinear(nn.Linear(embed_dim, embed_dim), r=r, alpha=alpha)
        self.k_proj = nn.Linear(embed_dim, embed_dim)  # normal
        self.v_proj = LoRALinear(nn.Linear(embed_dim, embed_dim), r=r, alpha=alpha)

        # W_o should NOT use LoRA
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape into heads
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = attn_scores.softmax(dim=-1)
        attn_output = attn_weights @ v

        # merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        return self.o_proj(attn_output)

mha = LoRAMultiHeadAttention(embed_dim=64, num_heads=4, r=4, alpha=8)

for name, p in mha.named_parameters():
    print(name, p.requires_grad)