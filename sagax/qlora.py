import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List

# NF4 table (from QLoRA / example)
NF4_TABLE = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7239560484886169, 1.0
], dtype=torch.float32)

class QLoRALinear(nn.Module):
    """
    QLoRA-style linear:
      - Block-wise NF4-like quantization of base weight (stored as packed uint8 + per-block scales)
      - LoRA adapters A (r, in) and B (out, r)
      - Forward: dequantize (cached) + base_out + scaling*(x @ A.T @ B.T)
    """
    BLOCK_SIZE = 64

    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 32, dropout: float = 0.0):
        super().__init__()
        # optional: support biasless only for simplicity
        assert base_linear.bias is None, "QLoRALinear currently requires base_linear.bias == None"

        self.out_features, self.in_features = base_linear.weight.shape
        self.r = r
        self.alpha = alpha
        self.scaling = float(alpha) / max(1, r)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # freeze base weights (we store quantized view)
        base_linear.weight.requires_grad = False

        # quantize base weight into packed bytes + per-block scale
        packed, scales_fp16, pad = self._quantize_nf4(base_linear.weight.data)
        # store as buffers so .to(device) works
        self.register_buffer("qweight", packed)   # uint8 packed (out, num_blocks * (BLOCK_SIZE//2))
        self.register_buffer("scales", scales_fp16)  # float16 (num_blocks,)
        self.register_buffer("pad", torch.tensor(int(pad), dtype=torch.int16))

        # LoRA adapters
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # caching for dequantized weight (float32) per-device
        self._w_deq_cache = None
        self._cache_device = None
        self._merged = False

    @staticmethod
    def _pad_to_block_dim(n: int, block_size: int) -> int:
        return (block_size - (n % block_size)) % block_size

    def _quantize_nf4(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Quantize w (out, in) into packed 4-bit indices per block along input dim.
        Returns:
            packed: uint8 tensor of shape (out, num_blocks * (BLOCK_SIZE//2))
            scales_fp16: float16 tensor of shape (num_blocks,)
            pad: int (number of padded input columns)
        """
        device = w.device
        w = w.float()
        out_f, in_f = w.shape

        pad = self._pad_to_block_dim(in_f, self.BLOCK_SIZE)
        if pad > 0:
            w = F.pad(w, (0, pad))  # pad input dim (right side)
        in_padded = w.shape[1]
        num_blocks = in_padded // self.BLOCK_SIZE

        # reshape into (out, num_blocks, BLOCK_SIZE)
        w_blocks = w.view(out_f, num_blocks, self.BLOCK_SIZE)  # (out, nb, bs)

        # transpose to (nb, out, bs)
        w_blocks_t = w_blocks.permute(1, 0, 2).contiguous()  # (nb, out, bs)
        absmax_block = w_blocks_t.abs().amax(dim=(1,2))  # (nb,)
        # scale mapping: NF4 roughly maps values to [-7,7], so scale = absmax / 7  => to avoid asymmetry issue
        scale = absmax_block / 7.0
        scale[scale == 0.0] = 1e-8

        # normalize each block: w_blocks_t / scale[:,None,None]
        w_norm = w_blocks_t / scale.view(-1, 1, 1)  # (nb, out, bs)

        # flatten and find nearest NF4 entry
        nf4 = NF4_TABLE.to(device)
        nb = num_blocks

        # compute distances and pick indices
        w_norm_flat = w_norm.view(nb, -1)  # (nb, out*bs)

        # (nb, out*bs, 16)
        dist = (w_norm_flat.unsqueeze(-1) - nf4.view(1,1,-1)).abs()
        indices = dist.argmin(dim=-1).view(nb, self.out_features, self.BLOCK_SIZE).to(torch.uint8)  # (nb, out, bs)

        # pack two 4-bit indices per byte along bs axis
        bs = self.BLOCK_SIZE
        if bs % 2 == 1:
            # pad last element to make even (unlikely for 64 but safe)
            pad_idx = torch.zeros(nb, self.out_features, 1, dtype=torch.uint8, device=device)
            indices = torch.cat([indices, pad_idx], dim=-1)
            bs += 1

        low = indices[:, :, 0:bs:2]   # (nb, out, bs/2)
        high = indices[:, :, 1:bs:2]  # (nb, out, bs/2)
        packed_per_block = (low | (high << 4)).contiguous()  # (nb, out, bs/2)
       
        packed_per_block = packed_per_block.permute(1, 0, 2).contiguous()  # (out, nb, bs/2)
        packed = packed_per_block.view(self.out_features, -1).to(torch.uint8)  # (out, nb * bs/2)

        scales_fp16 = scale.to(torch.float16)
        return packed, scales_fp16, pad

    @torch.no_grad()
    def _dequantize(self) -> torch.Tensor:
        """
        Dequantize packed representation into float32 weight (out, in).
        Uses cached version if device matches.
        """
        # If merged into cached float weight, return it
        device = self.qweight.device
        if self._w_deq_cache is not None and self._cache_device == device:
            return self._w_deq_cache

        packed = self.qweight  # (out, nb * half_bs)
        out_f, total_cols = packed.shape
        half_bs = (self.BLOCK_SIZE // 2)
        # number of blocks
        nb = total_cols // half_bs
        # reshape into (nb, out, half_bs)
        packed_blocks = packed.view(out_f, nb, half_bs).permute(1, 0, 2).contiguous()  # (nb, out, half_bs)

        # unpack low/high bit extraction
        low = (packed_blocks & 0x0F).to(torch.long)  # (nb, out, half_bs)
        high = ((packed_blocks >> 4) & 0x0F).to(torch.long)
        bs = half_bs * 2
        
        # interleave into indices shape (nb, out, bs)
        indices = torch.empty((nb, out_f, bs), dtype=torch.long, device=device)
        indices[:, :, 0:bs:2] = low
        indices[:, :, 1:bs:2] = high

        # map indices through NF4 table
        nf4 = NF4_TABLE.to(device)
        values = nf4[indices]  # (nb, out, bs) float32

        # multiply by per-block scale
        scales = self.scales.view(nb, 1, 1).to(device=device, dtype=torch.float32)  # (nb,1,1)
        w_blocks = values * scales  # (nb, out, bs)

        # flatten back to (out, nb*bs)
        w_blocks_t = w_blocks.permute(1, 0, 2).contiguous()  # (out, nb, bs)
        w_flat = w_blocks_t.view(out_f, nb * bs)  # (out, nb*bs)

        # trim padding
        pad = int(self.pad.item())
        if pad > 0:
            w_flat = w_flat[:, : (w_flat.shape[1] - pad)]

        weight = w_flat.to(torch.float32).contiguous()  # (out, in)
        # cache
        self._w_deq_cache = weight
        self._cache_device = device
        return weight

    def merge(self):
        """
        Merge LoRA delta into cached dequantized weight (float32).
        After merging, forward will use merged base+delta (no separate LoRA).
        """
        with torch.no_grad():
            if self._merged:
                return
            w = self._dequantize()  # float32
            delta = (self.lora_B @ self.lora_A) * self.scaling  # (out, in)
            w = w + delta
            self._w_deq_cache = w
            self._merged = True

    def unmerge(self):
        """Unmerge by clearing cache and force re-dequantize (base is still quantized)."""
        with torch.no_grad():
            if not self._merged:
                return
            self._w_deq_cache = None
            self._cache_device = None
            self._merged = False

    def quantization_mse(self, original_weight: torch.Tensor) -> float:
        with torch.no_grad():
            w_deq = self._dequantize().float()
            orig = original_weight.detach().float().to(w_deq.device)
            return float(((orig - w_deq) ** 2).mean().item())

    def get_lora_parameters(self) -> List[nn.Parameter]:
        return [self.lora_A, self.lora_B]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ensure dequantize on same device as x
        if self.qweight.device != x.device:
            self.qweight = self.qweight.to(x.device)
            self.scales = self.scales.to(x.device)
            self.pad = self.pad.to(x.device)
            # invalidate cache if device change
            self._w_deq_cache = None
            self._cache_device = None

        W = self._dequantize()
        base_out = x @ W.t()

        if self.r > 0 and not self._merged:
            lora_out = (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
            return base_out + lora_out
        elif self.r > 0 and self._merged:
            return base_out
        else:
            return base_out