import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        D: int  = len(data)
        optimizer = AdamW(model.parameters(), lr=lr) # For fun, rewrite this by hand

        for epoch in range(epochs):
            torch.manual_seed(epoch)
            ix = torch.randint(D - context_length, (batch_size,))
            x = torch.stack([data[i:i+context_length] for i in ix])
            y = torch.stack([data[i+1:i+1+context_length] for i in ix])

            logits = model(x)
            B,T,C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), y.view(B*T)) # For fun, rewrite this by hand

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return round(loss.item(), 4)