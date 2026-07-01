# coding: utf-8
"""加载 checkpoint，进行文本续写。"""
import argparse
from pathlib import Path

import torch

from transformer_lm import TransformerLM, decode, encode, get_device, load_vocab

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "checkpoints"


def load_model(ckpt_dir=CKPT_DIR):
    ckpt_dir = Path(ckpt_dir)
    vocab = load_vocab(ckpt_dir / "vocab.json")
    inv_vocab = {v: k for k, v in vocab.items()}
    payload = torch.load(ckpt_dir / "model.pt", map_location="cpu")
    cfg = payload["config"]
    device = get_device()
    model = TransformerLM(
        cfg["vocab_size"], cfg["d_model"], cfg["n_layers"],
        cfg["n_heads"], max_seq_len=cfg["max_seq_len"],
    )
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, vocab, inv_vocab, device


def generate(prompt, max_new_tokens=50, temperature=0.8, top_k=40, greedy=False):
    model, vocab, inv_vocab, device = load_model()
    ids = encode(prompt, vocab)
    x = torch.tensor([ids], device=device)
    if greedy:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = model(x[:, -model.max_seq_len:])[:, -1, :]
                x = torch.cat([x, logits.argmax(-1, keepdim=True)], dim=1)
        out_ids = x[0].tolist()
    else:
        out_ids = model.generate(x, max_new_tokens, temperature, top_k)[0].tolist()
    return prompt + decode(out_ids[len(ids):], inv_vocab)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="让他在半年之前，就不能做出")
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--greedy", action="store_true")
    args = p.parse_args()
    print(generate(args.prompt, args.max_new_tokens, args.temperature, args.top_k, args.greedy))
