import torch
from lora_mha import LoRAMultiHeadAttention  # your LoRAMHA module

# Hyperparameters
batch_size = 2
seq_len = 5
embed_dim = 64
num_heads = 4
r = 4
alpha = 8

# Create random input
x = torch.randn(batch_size, seq_len, embed_dim)

# Initialize LoRA MHA
mha = LoRAMultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, r=r, alpha=alpha)

# Forward pass
out = mha(x)

print("Input shape:", x.shape)
print("Output shape:", out.shape)

# Check which parameters are trainable
print("\nTrainable parameters:")
for name, p in mha.named_parameters():
    if p.requires_grad:
        print(name, p.shape)
