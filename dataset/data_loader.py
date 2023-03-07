import torch
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence

"""数据集处理和加载"""



class MyDataset(Dataset):
    """自定义数据集 句子+对应情感分类"""

    def __init__(self, sentences, emotions ):
        self.sentences = sentences
        self.emotion = emotions

    def __getitem__(self, index):
        return self.sentences[index]
        
    def __len__(self):
        return len(self.sentences)


"""数据处理 原始数据格式 PhraseId SentenceId Phrase	Sentiment"""

class glove_embedding():
    """将句子转化为对应矩阵表示"""
    def __init__(self, datas, glove_dict):
        self.dict_words = dict() ##句子中包含的单词的序号
        self.glove_word2vec_dict = glove_dict ##glove将单词编码成向量的对应字典
        self.word_dim = len(glove_dict.values[0]) ##glove中单词映射到向量的维度
        self.sentences_matrix = list() ##句子对应的矩阵表示
        datas.sort(key = lambda x:len(x[2].spilt()))  ##将数据按照词数的升序排列，以减少一个batch里不必要的padding操作
        
        self.emotions = [int(data[3]) for data in datas] ##数据第四列是同一行的句子对应的情感类别
        for data in datas:
            sentence = data[2] ##数据第三项是句子
            sentence.upper() ##将字母大写，以免因大小写将同一单词认定为是不同单词
            words = sentence.split() ##将句子分词
            sentence_matrix = [] ##用于存储句子对应的矩阵大小为n*word_dim，n为句子中词语个数，word_dim为glove将一个单词映为50维的矩阵
            for word in words:
                if word in self.glove_word2vec_dict:
                    sentence_matrix.append(self.glove_word2vec_dict[word]) ##在glove中寻找单词对应的向量
                else:
                    sentence_matrix.append(self.word_dim * [0]) ##若glove中无对应的向量，则为全零向量
            self.sentences_matrix.append(sentence) ##将单个句子添加到整个数据中


def collate_fn(batch_data):
    """调整数据的输出类型"""
    sentence, emotion = zip(*batch_data) ## zip(*batch_data)将数据打包成（x, y）形式
    sentences = [torch.LongTensor(sent) for sent in sentence]  # 把句子变成Longtensor类型
    padded_sents = pad_sequence(sentences, batch_first=True, padding_value=0)  # 自动padding
    return torch.LongTensor(padded_sents), torch.LongTensor(emotion)


def get_dataset_loader(sentences, emotions, batchsize):
    """创建dataloader供训练使用"""
    dataset = MyDataset(sentences, emotions) ##使用处理好的句子矩阵和对应情感类别构造Dataset
    dataloader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True, drop_last=True, collate_fn=collate_fn)
    return dataloader



            





