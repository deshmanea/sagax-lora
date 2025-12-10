import math
import torch
from torch import nn

class LoRALinear(nn.Module):

    """
        LoRA wrapper for nn.Linear.
        - base: the original linear layer (nn.Linear)
        - r: rank
        - alpha: scaling coefficient
        - merge_weights: when True, W has adapter merged (inference)
    """

    def __init__(self, base_linear:nn.Linear, r : int = 8 , alpha: int = 32,  dropout: float = 0.0 ):
        super().__init__()
        self.base = base_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for p in self.base.parameters():
            p.requires_grad = False

        if r > 0:
            # Wrap to make to matrix as learnable parameters
            self.A = nn.Parameter(torch.zeros(r, self.base.in_features))
            self.B = nn.Parameter(torch.zeros(self.base.out_features, r))

            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        else:
            # r == 0 means no adapter
            self.register_parameter('A', None)
            self.register_parameter('B', None)
        
        self.merged = False
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base output (always computed)

        base_out = self.base(x) # shape(barch, out)

        if self.r <= 0:
            return base_out

        if self.merged:
            # If merged, adapter already in base.weight
            return base_out

        #  compute LoRA delta
        x_drop = self.dropout(x)

        xA = x_drop @ self.A.T   # shape (batch, r)
        delta = xA @ self.B.T    # shape (batch, out)

        return base_out + self.scaling * delta
    
    
    def merge(self):
         '''Merge LoRA weights into base.weight for inference (in-place).'''
         
         with torch.no_grad():
            if self.r <= 0 or self.merged:
                return
            delta = (self.B @ self.A) * self.scaling
            self.base.weight.data += delta
            self.merged = True


    def unmerge(self):
            '''Revert merge: subtract out the LoRA contribution.'''

            with torch.no_grad():
                if self.r <= 0 or not self.merged:
                    return
                delta = (self.B @ self.A) * self.scaling
                self.base.weight.data -= delta
                self.merged = False

                
    def train(self, mode = True):
        super().train(mode)
        return self


    def get_lora_parameters(self):
        return [self.A, self.B]