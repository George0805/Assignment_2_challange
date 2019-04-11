

# Importing libraries
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import random
import string
import time, math


# 计算字符种类的数量
all_characters = string.printable
n_characters = len(all_characters)

# txt文档
with open('/Users/George 1/OneDrive/课件/Leiden/NN/2018-re/assignment 2/data/shakespeare.txt') as f:
    text = f.read()

# 计算所有字符数量
file_len = len(text)

# 设定每次训练的batch size
chunk_len = 200

# 从txt文档中提取一个batch作为训练数据
def random_chunk():
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return text[start_index:end_index]


# 字符转化为数字
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor)


# 制作训练集，target就是label，我们要用每次的从1到t的字符预测t+1的字符
def random_training_set():
    chunk = random_chunk()
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target




# define to encode one-hot labels
# def one_hot_encode(encoded):
#
#     # binary encode
#     onehot_encoder = OneHotEncoder(sparse=False)
#     integer_encoded = encoded.reshape(len(encoded), 1)
#     one_hot = onehot_encoder.fit_transform(integer_encoded)
#
#     return one_hot

# # invert back??
# def one_hot_invert(one_hot):
#     label_encoder = LabelEncoder()
#     integer_encoded = label_encoder.fit_transform(values)
#     # invert
#     inverted = label_encoder.inverse_transform([argmax(one_hot[0, :])])
#
#     return inverted



# a=one_hot_encode(encoded)
#
# b=torch.from_numpy(a)
#
# train_data = b[:5058199]
# test_data = b[5058199:5458199]
#
# train_loader = Data.DataLoader(dataset=b, batch_size=10, shuffle=True, num_workers=2)



# class RNN(nn.Module):
#     def __init__(self):
#         super(RNN, self).__init__()
#
#         self.rnn = nn.LSTM(
#             input_size=1,
#             hidden_size=64,
#             num_layers=1,
#             batch_first=True,
#         )
#         self.out = nn.Linear(64, 10)
#
#     def forward(self, x):
#         r_out, (h_n, h_c) = self.rnn(x, None)
#         out = self.out(r_out[:, -1, :])
#         return out
#
# rnn = RNN()
# print(rnn)


# 做RNN模型。用的是GRU，暂时还没加LSTM
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))


# 用训练好的模型产生一批字符，默认开头字符是'A'
def evaluate(prime_str='A', predict_len=100, temperature=0.8):
    hidden = decoder.init_hidden()
    prime_input = char_tensor(prime_str)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[p], hidden)
    inp = prime_input[-1]

    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)

    return predicted



# 计时器
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# 训练函数，criterion是损失函数
def train(inp, target):
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0

    for c in range(chunk_len):
        output, hidden = decoder(inp[0][c], hidden)
        loss += criterion(output, target[0][c].unsqueeze(0))

    loss.backward()
    decoder_optimizer.step()

    return loss.data / chunk_len


# 初参数
n_epochs = 200
print_every = 10
plot_every = 10
hidden_size = 100
n_layers = 1
lr = 0.005

# 开始定义RNN，优化函数、损失函数
decoder = RNN(n_characters, hidden_size, n_characters, n_layers)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# 开始训练
start = time.time()
all_losses = []
loss_avg = 0

for epoch in range(1, n_epochs + 1):
    data = random_training_set()
    loss = train(data[0].unsqueeze(0), data[1].unsqueeze(0))
    loss_avg += loss

    if epoch % print_every == 0:
        print('[Total time: %s (iter: %d  per: %d%%) Loss: %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
        print(evaluate('Wh', 100), '\n')  # 设定开头字符是'Wh'

    if epoch % plot_every == 0:
        all_losses.append(loss_avg / plot_every)
        loss_avg = 0









