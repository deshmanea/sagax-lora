import torch
from torch import nn
from qlora import QLoRALinear

torch.manual_seed(0)
in_f = 130  # intentionally not multiple of 64 to test padding
out_f = 128

base = nn.Linear(in_f, out_f, bias=None)
ql = QLoRALinear(base, r=8, alpha=32)

# small MSE check
mse = ql.quantization_mse(base.weight)
print("Quantization MSE:", mse)

# forward sanity
x = torch.randn(3, in_f)
y = ql(x)
print("Output shape:", y.shape)

# ensure LoRA params are trainable and base is frozen
for name, p in ql.named_parameters():
    print(name, p.requires_grad, p.shape)
