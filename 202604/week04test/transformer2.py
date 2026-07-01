import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, num_hidden, num_heads):
        super().__init__()
        assert num_hidden % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = num_hidden // num_heads

        self.qkv = nn.Linear(num_hidden, num_hidden * 3)
        self.out = nn.Linear(num_hidden, num_hidden)

    def forward(self, x, mask=None):
        '''
        Attention(Q*K.T / sqrt(num_hidden) * V)
        :param x:
        :param mask:
        :return:
        '''
        B, T, H = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = nn.functional.softmax(scores, dim=-1) @ v

        attention = attention.transpose(1, 2).contiguous().view(B, T, H)

        return self.out(attention)

class EncoderLayer(nn.Module):
    def __init__(self, num_hidden, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(num_hidden, num_heads)
        self.ln1 = nn.LayerNorm(num_hidden)
        self.ffn = nn.Sequential(
            nn.Linear(num_hidden, num_hidden * 4),
            nn.GELU(),
            nn.Linear(num_hidden * 4, num_hidden),
        )
        self.ln2 = nn.LayerNorm(num_hidden)

    def forward(self, x, mask=None):
        x = self.ln1(x + self.attn(x, mask))
        x = self.ln2(x + self.ffn(x))
        return x

class TransformerLayers(nn.Module):
    def __init__(self, num_layers=12, num_hidden=768, num_heads=12):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(num_hidden, num_heads) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

if __name__ == "__main__":
    x = torch.randn(2, 16, 768)

    model = TransformerLayers(num_layers=12, num_hidden=768, num_heads=12)
    print(model(x).shape)





