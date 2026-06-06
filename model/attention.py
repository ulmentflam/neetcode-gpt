import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    __constants__ = ["d_model", "d_head"]
    d_model: int
    d_head: int


    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.d_model = embedding_dim
        self.d_head = attention_dim
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        # Instead of creating 3 linear projections, we create one that holds (K, Q, V) that we can split.
        self.in_proj = nn.Linear(embedding_dim, 3 * attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        x: TensorType[float] = self.in_proj(embedded)
        K, Q, V = torch.chunk(x, 3, dim=-1)
        B, T, C = Q.shape
        d_k = K.shape[-1]
        # (B, T, C) @ (B, C, T) -> B, T, T
        scores = Q @ K.transpose(-1, -2) / math.sqrt(d_k)
        mask = torch.tril(torch.ones(T, T)) == 0
        scores = scores.masked_fill(mask, float('-inf'))
        scores = F.softmax(scores, dim=-1)
        return torch.round(scores @ V, decimals=4)
