import json
import random

import numpy as np
import torch
from torch import nn


class RNNClassifierModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(RNNClassifierModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # self.rnn = nn.RNN(embedding_dim, hidden_size, bias=False, batch_first=True)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, bias=False, batch_first=True)
        self.layerNorm = nn.Linear(hidden_size, output_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        embedded = self.embedding(x)
        output, h_n = self.rnn(embedded)
        y_hat = output[:, -1, :]
        y_hat = self.layerNorm(y_hat)
        if y is not None:
            return self.loss(y_hat, y)
        return y_hat


def build_vocab():
    # chars = "abcdefgh"  # 字符集
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab


def build_simple(vocab, sentences_len):
    # 去掉pad和unk
    validate_char = [c for c in list(vocab.keys()) if c not in ['pad', 'unk']]
    # 生成样本长度个字符
    x_chars = random.choices(validate_char, k=sentences_len)
    # 随机生成一个目标字符
    target_chars = ['a']
    target_chars = ['a', 'b', 'c']
    # 记录目标所在位置
    positions = [idx for idx, char in enumerate(x_chars) if char in target_chars]
    # 计算标签（0~4最大的，5未出现）
    y = max(positions) if positions else sentences_len
    # s -> n
    x = [vocab.get(c, vocab['unk']) for c in x_chars]

    return x, y


def build_dataset(vocab, batch_size, sentences_len):
    inputs = []
    labels = []
    for i in range(batch_size):
        x, y = build_simple(vocab, sentences_len)
        inputs.append(x)
        labels.append(y)
    return torch.LongTensor(inputs), torch.LongTensor(labels)


def evaluate(model, vocab, sentences_len):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(vocab, test_sample_num, sentences_len)
    correct = 0
    with torch.no_grad():
        y_pred_logits = model(x)
        y_pred = torch.argmax(y_pred_logits, dim=1)

        correct = (y_pred == y).sum().item()

    accuracy = correct / test_sample_num
    # print(f"测试集总样本：{test_sample_num}个，正确预测：{correct}个，准确率：{accuracy:.4f}")
    return accuracy


def main():
    num_epochs = 20  # 训练轮数
    batch_size = 20  # 每次训练的样本数
    train_sample_num = 2000  # 每轮总共训练样本数
    embedding_dim = 16  # 每个字的维度
    hidden_size = 32  # RNN影藏层维度
    output_size = 6  # 输出类别数（0~5对应1~6）
    sentences_len = 5  # 样本的文本长度
    learning_rate = 1e-3  # 学习率
    # 构建词表
    vocab = build_vocab()
    # 初始化模型
    model = RNNClassifierModel(len(vocab), embedding_dim, hidden_size, output_size)
    # Adam优化器实现简单，计算高效，对内存的需求少
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        watch_loss = []
        for _ in range(train_sample_num // batch_size):
            x, y = build_dataset(vocab, batch_size, sentences_len)
            optimizer.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重

            watch_loss.append(loss.item())

        accuracy = evaluate(model, vocab, sentences_len)
        print('第%d轮，平均loss：%f，acc: %f' % (epoch + 1, np.mean(watch_loss), accuracy))

        # 保存训练数据
    torch.save(model.state_dict(), 'model.pt')
    # 保存词表
    with open('vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)


def predict(model_path, vocab_path, test_strs):
    vocab = json.load(open(vocab_path, 'r', encoding='utf-8'))
    embedding_dim = 16
    hidden_size = 32
    output_size = 6
    sentence_length = 5
    model = RNNClassifierModel(len(vocab), embedding_dim, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path))
    x = []
    for test_str in test_strs:
        # 截取前sentence_length个字符，不足则补pad（这里简化为直接截断）
        x_str = test_str[:sentence_length]
        x.append([vocab.get(c, vocab['unk']) for c in x_str])

    model.eval()
    with torch.no_grad():
        y_pred = model(torch.LongTensor(x))
        y_pred = torch.argmax(y_pred, dim=1)  # 取预测类别
    for i, s in enumerate(test_strs):
        pred_class = y_pred[i].item()  # 类别从0→1映射
        print(f'输入：{s:8}，预测类别：{pred_class}（{"无" if pred_class == 5 else "第" + str(pred_class + 1) + "位"}）')


if __name__ == '__main__':
    # main()
    test_strings = ['abcdefg', 'lkaajhy', 'hddhqw', 'fhxceui', 'ppppppp', 'aaaaa']
    predict('model.pt', 'vocab.json', test_strings)
