
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




# txt文档
with open('/Users/George 1/OneDrive/课件/Leiden/NN/2018-re/assignment 2/data/shakespeare.txt') as f:
    text = f.read()

# 计算唯一单词数量
all_words = set(text.split())
n_words = len(all_words)

# 计算所有单词数量
file_len = len(text.split())

# 设定每次训练的batch size
chunk_len = 200

# 从txt文档中提取一个batch作为训练数据
def random_chunk():
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return text.split()[start_index:end_index]


# index vocabulary
word_to_ix = {word: i for i, word in enumerate(all_words)}


# word to index vector
def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return Variable(torch.tensor(idxs, dtype=torch.long))


# make training set
def random_training_set():
    chunk = random_chunk()
    inp = make_context_vector(chunk[:-1], word_to_ix)
    target = make_context_vector(chunk[1:], word_to_ix)
    return inp, target


# 做RNN模型。用LSTM
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        # self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        # output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output, hidden = self.lstm(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        #return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        h_0 = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        c_0 = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        out = (h_0, c_0)
        return out


# 用训练好的模型产生一批字符，默认开头单词是'Much'
def evaluate(prime_word=['Much', 'is'], predict_len=15, temperature=0.8):
    hidden = decoder.init_hidden()
    prime_input = make_context_vector(prime_word, word_to_ix)
    predicted = prime_word

    # Use priming string to "build up" hidden state
    for p in range(len(prime_word) - 1):
        _, hidden = decoder(prime_input[p], hidden)
    inp = prime_input[-1]

    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted word to string and use as next input
        predicted_word = [name for name, ind in word_to_ix.items() if ind == top_i]
        predicted.append(*predicted_word)
        inp = make_context_vector(predicted_word, word_to_ix)

    return " ".join(predicted)



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

    for c in range(chunk_len-1):
        output, hidden = decoder(inp[0][c], hidden)
        loss += criterion(output, target[0][c].unsqueeze(0))

    loss.backward()
    decoder_optimizer.step()

    return loss.data / chunk_len


# 初参数
n_epochs = 10000
print_every = 10
plot_every = 10
hidden_size = 100
n_layers = 1
lr = 0.005

# 开始定义RNN，优化函数、损失函数
decoder = RNN(n_words, hidden_size, n_words, n_layers)
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
        print(evaluate(prime_word=['Much', 'is'], predict_len=20), '\n')  # 设定开头字符

    if epoch % plot_every == 0:
        all_losses.append(loss_avg / plot_every)
        loss_avg = 0









