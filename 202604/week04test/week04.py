from pathlib import Path

import torch
from torch import nn
from transformers import BertModel

# 必须用绝对路径，新版 huggingface_hub 不接受 ../bert-base-chinese
BERT_DIR = str(Path(__file__).resolve().parent.parent / "bert-base-chinese")
bert = BertModel.from_pretrained(BERT_DIR, return_dict=False, local_files_only=True)
state_dict = bert.state_dict()

print("====== BERT 模型参数 ======")
total = 0
for key, param in bert.state_dict().items():
    param_num = param.numel()
    total += param_num
    print(f"{key:<60} {list(param.shape)!s:<18} {param_num:,}")
print("=" * 100)
print(f"参数总量: {total:,}")


class BertModelWithTorch(nn.Module):
    def __init__(self, state_dict):
        super(BertModelWithTorch, self).__init__()

        self.num_layers = 12  # bert-base 是 12 层！你写 6 会错！
        self.hidden_size = 768
        self.num_attention_heads = 12

        # Embedding 层
        self.word_embeddings = nn.Embedding.from_pretrained(state_dict['embeddings.word_embeddings.weight'])
        self.position_embeddings = nn.Embedding.from_pretrained(state_dict['embeddings.position_embeddings.weight'])
        self.token_type_embeddings = nn.Embedding.from_pretrained(state_dict['embeddings.token_type_embeddings.weight'])

        self.word_embeddings.requires_grad_(False)
        self.position_embeddings.requires_grad_(False)
        self.token_type_embeddings.requires_grad_(False)

        # LayerNorm
        self.emb_layer_norm = nn.LayerNorm(self.hidden_size)
        self.emb_layer_norm.load_state_dict({
            "weight": state_dict['embeddings.LayerNorm.weight'],
            "bias": state_dict['embeddings.LayerNorm.bias']
        })

        # Transformer 层
        self.transformer_layers = nn.ModuleList()
        for idx in range(self.num_layers):
            self.transformer_layers.append(
                TransformerLayers(state_dict, idx, self.hidden_size, self.num_attention_heads)
            )

        # Pooler
        self.pooler = nn.Linear(self.hidden_size, self.hidden_size)
        self.pooler.weight.data = state_dict["pooler.dense.weight"]
        self.pooler.bias.data = state_dict["pooler.dense.bias"]

    def forward(self, x):
        x_embedding = self.embedding_forward(x)
        sequence_output = self.transformer_forward(x_embedding)
        pooler_output = torch.tanh(self.pooler(sequence_output[:, 0:1, :].squeeze(1)))
        return sequence_output, pooler_output

    def transformer_forward(self, x):
        for layer in self.transformer_layers:
            x = layer(x)
        return x

    def embedding_forward(self, x):
        we = self.word_embeddings(x)
        position_ids = torch.arange(we.shape[1], device=x.device)[None, :]
        pe = self.position_embeddings(position_ids)
        token_type_ids = torch.zeros_like(x)
        te = self.token_type_embeddings(token_type_ids)

        embedding = we + pe + te
        embedding = self.emb_layer_norm(embedding)
        return embedding


class TransformerLayers(nn.Module):
    def __init__(self, state_dict, layer_idx, hidden_size, num_attention_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        # QKV
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.query.load_state_dict({
            "weight": state_dict[f'encoder.layer.{layer_idx}.attention.self.query.weight'],
            "bias": state_dict[f'encoder.layer.{layer_idx}.attention.self.query.bias']
        })
        self.key.load_state_dict({
            "weight": state_dict[f'encoder.layer.{layer_idx}.attention.self.key.weight'],
            "bias": state_dict[f'encoder.layer.{layer_idx}.attention.self.key.bias']
        })
        self.value.load_state_dict({
            "weight": state_dict[f'encoder.layer.{layer_idx}.attention.self.value.weight'],
            "bias": state_dict[f'encoder.layer.{layer_idx}.attention.self.value.bias']
        })

        # Attention Output
        self.attention_output = nn.Linear(hidden_size, hidden_size)
        self.attention_output.load_state_dict({
            "weight": state_dict[f'encoder.layer.{layer_idx}.attention.output.dense.weight'],
            "bias": state_dict[f'encoder.layer.{layer_idx}.attention.output.dense.bias']
        })

        # Attention LayerNorm
        self.attention_layer_norm = nn.LayerNorm(hidden_size)
        self.attention_layer_norm.load_state_dict({
            "weight": state_dict[f'encoder.layer.{layer_idx}.attention.output.LayerNorm.weight'],
            "bias": state_dict[f'encoder.layer.{layer_idx}.attention.output.LayerNorm.bias']
        })

        # FFN
        self.intermediate = nn.Linear(hidden_size, 4 * hidden_size)
        self.intermediate.load_state_dict({
            "weight": state_dict[f'encoder.layer.{layer_idx}.intermediate.dense.weight'],
            "bias": state_dict[f'encoder.layer.{layer_idx}.intermediate.dense.bias']
        })

        self.gelu = nn.GELU()

        self.output = nn.Linear(4 * hidden_size, hidden_size)
        self.output.load_state_dict({
            "weight": state_dict[f'encoder.layer.{layer_idx}.output.dense.weight'],
            "bias": state_dict[f'encoder.layer.{layer_idx}.output.dense.bias']
        })

        # Output LayerNorm
        self.output_layer_norm = nn.LayerNorm(hidden_size)
        self.output_layer_norm.load_state_dict({
            "weight": state_dict[f'encoder.layer.{layer_idx}.output.LayerNorm.weight'],
            "bias": state_dict[f'encoder.layer.{layer_idx}.output.LayerNorm.bias']
        })

    def forward(self, x):
        attn_out = self.self_attention(x)
        x = self.attention_layer_norm(x + attn_out)

        ffn_out = self.feed_forward(x)
        x = self.output_layer_norm(x + ffn_out)
        return x

    def feed_forward(self, x):
        x = self.intermediate(x)
        x = self.gelu(x)
        x = self.output(x)
        return x

    def self_attention(self, x):
        B, S, H = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # 多头拆分
        q = q.view(B, S, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        k = k.view(B, S, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        v = v.view(B, S, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # 注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float32))
        attn_probs = nn.functional.softmax(attn_scores, dim=-1)

        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(B, S, self.hidden_size)

        out = self.attention_output(context)
        return out


# ===================== 测试 =====================
x = torch.LongTensor([[873, 1963, 705, 1745]])

db_pytorch = BertModelWithTorch(state_dict)
db_pytorch.eval()

with torch.no_grad():
    pytorch_sequence_output, pytorch_pooler_output = db_pytorch(x)

print("PyTorch 实现 - Sequence Output Shape:", pytorch_sequence_output.shape)
print("PyTorch 实现 - Pooler Output Shape:", pytorch_pooler_output.shape)

print("\n" + "=" * 50)
print("Hugging Face 对比")

bert.eval()
with torch.no_grad():
    hf_sequence_output, hf_pooler_output = bert(x)

print("HF - Sequence Output Shape:", hf_sequence_output.shape)
print("HF - Pooler Output Shape:", hf_pooler_output.shape)

print("\n差异 L2 Norm:", torch.norm(pytorch_pooler_output - hf_pooler_output).item())