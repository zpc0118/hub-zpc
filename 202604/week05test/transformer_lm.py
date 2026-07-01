# coding: utf-8
"""单向 Transformer 字符级语言模型：训练并保存 checkpoint。"""
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.d_k).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(self.causal_mask[:, :, :t, :t] == 0, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.out(out)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_seq_len, dropout):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.ln1(x + self.dropout(self.attn(x)))
        x = self.ln2(x + self.ffn(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=4, d_ff=512,
                 max_seq_len=64, dropout=0.1, pad_id=0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff, max_seq_len, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, idx):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.max_seq_len:])[:, -1, :]
            logits = logits / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def load_corpus(path):
    for enc in ("utf-8", "gbk", "utf-8-sig"):
        try:
            return Path(path).read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def build_vocab(corpus):
    vocab = {"<pad>": 0, "<unk>": 1}
    for i, ch in enumerate(sorted(set(corpus)), start=2):
        vocab[ch] = i
    return vocab


def save_vocab(vocab, path):
    path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")


def load_vocab(path):
    return json.loads(path.read_text(encoding="utf-8"))


def encode(text, vocab):
    unk = vocab["<unk>"]
    return [vocab.get(ch, unk) for ch in text]


def decode(ids, inv_vocab):
    return "".join(inv_vocab.get(i, "<unk>") for i in ids)


def sample_batch(corpus, vocab, batch_size, seq_len):
    need = seq_len + 1
    xs, ys = [], []
    for _ in range(batch_size):
        start = random.randint(0, len(corpus) - need)
        ids = encode(corpus[start:start + need], vocab)
        xs.append(ids[:-1])
        ys.append(ids[1:])
    return torch.tensor(xs), torch.tensor(ys)


def train(epochs=10, batch_size=64, seq_len=64, steps_per_epoch=200, lr=3e-4):
    corpus_path = HERE / "corpus.txt"
    if not corpus_path.is_file():
        raise FileNotFoundError(f"请把语料放到: {corpus_path}")

    save_dir = HERE / "checkpoints"
    save_dir.mkdir(exist_ok=True)

    corpus = load_corpus(corpus_path)
    vocab = build_vocab(corpus)
    inv_vocab = {v: k for k, v in vocab.items()}
    save_vocab(vocab, save_dir / "vocab.json")

    device = get_device()
    model = TransformerLM(len(vocab), max_seq_len=seq_len).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"语料: {corpus_path} ({len(corpus)} 字)  设备: {device}")

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            x, y = sample_batch(corpus, vocab, batch_size, seq_len)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())

        print(f"epoch {epoch}/{epochs}  loss={np.mean(losses):.4f}")

        model.eval()
        prompt = "黄金"
        p = torch.tensor([encode(prompt, vocab)], device=device)
        out = model.generate(p, 30, temperature=0.8, top_k=40)
        print("样例:", prompt + decode(out[0].tolist()[len(prompt):], inv_vocab))

    torch.save({
        "model_state": model.state_dict(),
        "config": {"vocab_size": len(vocab), "d_model": 128, "n_layers": 4,
                   "n_heads": 4, "max_seq_len": seq_len},
    }, save_dir / "model.pt")
    print(f"已保存: {save_dir / 'model.pt'}")


if __name__ == "__main__":
    train()
