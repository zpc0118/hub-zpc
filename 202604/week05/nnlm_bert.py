import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

"""
基于pytorch的BERT语言模型
"""

PRETRAIN_MODEL_PATH = str(Path(__file__).resolve().parent.parent / "bert-base-chinese")
tokenizer = BertTokenizer.from_pretrained(PRETRAIN_MODEL_PATH, local_files_only=True)


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab_size):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)

        self.bert = BertModel.from_pretrained(PRETRAIN_MODEL_PATH, local_files_only=True)
        # BERT的隐藏层维度是768，需要适配到input_dim
        hidden_size = self.bert.config.hidden_size
        self.bert_proj = nn.Linear(hidden_size, input_dim)  # if hidden_size != input_dim else nn.Identity()

        self.classify = nn.Linear(input_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # 使用BERT替代原来的embedding+LSTM
        # x shape: (batch_size, sen_len) - 自定义vocab的token_ids
        # 注意：BERT的embedding层需要正确的token_ids，这里我们使用BERT的embedding层
        # 为了兼容自定义vocab，我们使用BERT的embedding层来获取字符表示
        # 创建attention_mask（非padding位置为1）
        attention_mask = (x != 0).long()

        # 使用BERT的embedding层获取字符embedding
        # 注意：这里假设自定义vocab的token_ids在BERT的vocab范围内，或者使用[UNK] token
        # 为了安全，我们将超出范围的token_ids映射到[UNK] (BERT的[UNK] token_id通常是100)
        bert_vocab_size = self.bert.config.vocab_size
        unk_token_id = tokenizer.unk_token_id if hasattr(tokenizer, "unk_token_id") else 100
        bert_input_ids = torch.where(x < bert_vocab_size, x, torch.full_like(x, unk_token_id))

        # BERT前向传播
        bert_outputs = self.bert(input_ids=bert_input_ids, attention_mask=attention_mask)
        # 处理BERT输出的两种格式：元组或对象
        if isinstance(bert_outputs, tuple):
            x = bert_outputs[0]  # 元组格式：第一个元素是last_hidden_state
        else:
            x = bert_outputs.last_hidden_state  # 对象格式：直接访问属性
        # x shape: (batch_size, sen_len, bert_hidden_size)
        # 投影到input_dim维度
        x = self.bert_proj(x)  # output shape:(batch_size, sen_len, input_dim)
        x = self.dropout(x)

        y_pred = self.classify(x)  # output shape:(batch_size, sen_len, vocab_size)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)


# 加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>": 0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]  # 去掉结尾换行符
            vocab[char] = index + 1  # 留出0位给pad token
    return vocab


# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位
    # print(window, target)
    x = [vocab.get(word, vocab["<UNK>"]) for word in window]  # 将字转换成序号
    y = [vocab.get(word, vocab["<UNK>"]) for word in target]
    return x, y


# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# window_size 样本长度
# corpus 语料字符串
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim):
    vocab_size = len(vocab)
    model = LanguageModel(char_dim, vocab_size)
    return model


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# 文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
            x = torch.LongTensor([x])
            device = get_device()
            x = x.to(device)
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = reverse_vocab[index]
    return openings


def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


# 计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["<UNK>"]) for char in window]
            x = torch.LongTensor([x])
            device = get_device()
            x = x.to(device)
            target = sentence[i]
            target_index = vocab.get(target, vocab["<UNK>"])
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * (-1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 64  # 每次训练样本个数
    train_sample = 50000  # 每轮训练总共训练的样本总数
    char_dim = 256  # 每个字的维度
    window_size = 10  # 样本文本长度
    vocab = build_vocab("vocab.txt")  # 建立字表
    corpus = load_corpus(corpus_path)  # 加载语料

    model = build_model(vocab, char_dim)  # 建立模型
    device = get_device()
    model = model.to(device)
    print(f"使用设备: {device}")
    optim = torch.optim.Adam(model.parameters(), lr=0.01)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, window_size, corpus)  # 构建一组训练样本
            device = get_device()
            x, y = x.to(device), y.to(device)
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    train("corpus.txt", False)
