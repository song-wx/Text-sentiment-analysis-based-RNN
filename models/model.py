import torch
import torch.nn as nn
import torch.functional as F


class My_RNN(nn.Module):
    
    """定义RNN网络结构，包括一个隐藏层，一个全连接层"""
    def __init__(self, input_len, hidden_len, words_num, output_len, weight, layer_num = 1, nonlinearity = 'tanh', 
                 batch_first=True, dropout = 0.5):
        super(My_RNN, self).__init__()
        self.input_len = input_len ##输入的维度，即词嵌入的维度
        self.words_num = words_num ##输入的句子中词的个数
        self.hidden_len = hidden_len ##隐藏层的维度
        self.layer_num = layer_num ##隐藏层的数量，默认值为1
        self.output_len = output_len ##输出的维度，即类别个数
        self.dropout = nn.Dropout(dropout) ##dropout层
        self.embedding = nn.Embedding( num_embeddings=self.words_num, embedding_dim=self.input_len, _weight=weight).cuda() ##对输入的句子做词嵌入
        ##定义rnn层
        self.rnn = nn.RNN(input_size=self.input_len, hidden_size=self.hidden_len, num_layers=self.layer_num, nonlinearity=nonlinearity,
                          batch_first=batch_first, dropout = dropout).cuda()
        ##定义全连接层
        self.fc = nn.Linear(in_features=self.hidden_len, out_features=self.output_len, bias=True).cuda()


    def forward(self, x):
        """x：数据，维度是[batchsize, 句子单词个数]"""
        x = torch.LongTensor(x).cuda()
        batchsize = x.size(0) ##获取batchsize
        """output: 词嵌入之后维度为[batchsize, 句子单词个数，词嵌入的维度]"""
        output = self.embedding(x) ##进行词嵌入，得到的输出维度为[batchsize, 句子单词个数， 词嵌入的dim]
        output = self.dropout(output) ##以dropout的概率变为0，以实现后续网络在更新参数时忽略掉部分参数

        ##初始化h0为全零向量,即初始的隐状态为零
        h0 = torch.randn(self.layer_num, batchsize, self.hidden_len).cuda()
        """dropout后不变，经过隐藏层后，只取最后一个时刻的隐状态输出，维度为[1，batch_size, hidden_len]"""
        _, hn = self.rnn(output, h0) 
        """经过全连接层后，输出维度为[1，batch_size, output_len]，之后压缩掉第0维度，得到的输出维度[batch_size, output_len]"""
        output = self.fc(hn).squeeze(0)
        return output


