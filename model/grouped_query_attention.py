import torch
import torch.nn as nn
from torchtyping import TensorType

class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, D = x.shape
        # 1. Project x into Q, K, V using the projection layers
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 2. Reshape into heads: Q has num_heads, K and V have num_kv_heads
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, T, head_dim]
        K = K.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2) # [B, num_kv_heads, T, head_dim]
        V = V.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2) # [B, num_kv_heads, T, head_dim]
        
        # 3. Expand K, V by repeating each KV head (num_heads // num_kv_heads) times
        repeat = self.num_heads // self.num_kv_heads
        K = K.repeat_interleave(repeat, dim=1)
        V = V.repeat_interleave(repeat, dim=1)
        
        # 4. Compute scaled dot-product attention with causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device))
        d_k = K.shape[-1]
        scores = Q @ K.transpose(-2, -1)
        scores *= d_k ** -0.5
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        out = weights @ V

        # 5. Concatenate heads and apply output projection
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.output_proj(out)
        # 6. Return rounded output (decimals=4)
        return torch.round(out, decimals=4)

