import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_head):
        super().__init__()
        assert hidden_size % num_head == 0
        self.num_head = num_head
        self.head_dim = hidden_size // num_head

        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, mask=None):
        B, L, D = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B, L, self.num_head, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_head, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_head, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1) @ v

        out = attn.transpose(1, 2).contiguous().view(B, L, D)

        return self.out(out)


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_head, ff):
        super().__init__()

        self.attn = MultiHeadAttention(hidden_size, num_head)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ffLayer = nn.Sequential(
            nn.Linear(hidden_size, ff),
            nn.GELU(),
            nn.Linear(ff, hidden_size),
        )
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        x = self.ln1(x + self.attn(x, mask))
        x = self.ln2(x + self.ffLayer(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size=768, num_head=12, num_layer=12):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                EncoderLayer(hidden_size, num_head, hidden_size * 4) for _ in range(num_layer)
            ]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


if __name__ == "__main__":
    hidden = 768
    num_head = 12
    num_layer = 12

    model = TransformerEncoder(hidden, num_layer, num_head)
    x = torch.randn(2, 12, 768)
    print(model(x))
