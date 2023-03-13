import torch
import csv
import random
from dataset.data_loader import *
from models.model import My_RNN
from train import train
import matplotlib.pyplot as plt

def main():
    
    ##读取训练和测试数据
    with open('dataset/train.tsv') as f:
        tsvreader = csv.reader(f,delimiter='\t')
        train_list = list(tsvreader)

    #由于kaggle上的测试集未给出sentiment label，故采用在训练集中以一定比例分出部分当测试集
    # with open('dataset/test.tsv') as f:
    #     tsvreader = csv.reader(f,delimiter='\t')
    #     test_list = list(tsvreader)

    data = train_list[1:] ##原先的第一个数据是数据的类型，PhraseId	SentenceId	Phrase	Sentiment
    # train_data, test_data = data_split(data, test_rate=0.3)

    ##读取Glove编码表
    with open('dataset/glove.6B.50d.txt','rb') as f:  # for glove embedding
        lines=f.readlines()
    
    ##将Glove编码表转化问单词->向量的字典
    glove_dict = dict()
    n=len(lines)
    for i in range(n):
        line=lines[i].split()
        glove_dict[line[0].decode("utf-8").upper()]=[float(line[j]) for j in range(1,51)]

    ##初始化训练参数
    epoch_num = 50 ##eopch数
    learning_rate = 0.0005 ##学习率
    batchsize = 256 
    input_len = 50 ##输入的维度，即词嵌入的维度
    hidden_len = 50 ##隐藏层的维度
    output_len = 5 ##输出的维度，即情感分类的个数，此数据集为5
    test_rate = 0.3 ##训练集与测试集7：3划分

    data_processed = glove_embedding(data, glove_dict, test_rate) ##构造了单词到序号的映射，和序号到词向量的映射（用于nn.embedding中的weight），和句子中单词的序号
    
    """构造训练集和测试集的dataloader"""
    train_loader = get_dataset_loader(data_processed.test_sentences_ID, data_processed.test_sentiments, batchsize=batchsize)
    test_loader = get_dataset_loader(data_processed.train_sentences_ID, data_processed.train_sentiments, batchsize=batchsize)

    sentiment_analysis_model = My_RNN(50, 50, data_processed.word_nums, output_len=output_len,
                                       weight=torch.tensor(data_processed.embedding_list, dtype=torch.float), layer_num=2)
    
    """开始训练，返回损失和精度情况"""
    train_loss_record, test_loss_record, train_acc_record, test_acc_record = train(sentiment_analysis_model, trainloader=train_loader, 
                                                                                   testloader=test_loader, epoch_nums=epoch_num, 
                                                                                   learning_rate=learning_rate,)

    """画图"""
    x = list(range(1,epoch_num + 1))
    ##损失
    plt.subplot(1, 2, 1)
    plt.plot(x, train_loss_record, 'r--', label='Train loss')
    plt.plot(x, test_loss_record, 'g-', label="Test loss")
    plt.legend()
    plt.legend(fontsize=10)
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    ##精度
    plt.subplot(1, 2, 2)
    plt.plot(x, train_acc_record, 'b--', label='Train accuracy')
    plt.plot(x, test_acc_record, 'y--', label='Test accuracy')
    plt.legend()
    plt.legend(fontsize=10)
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    #保存
    plt.ylim(0, 1)
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(8, 8, forward=True)
    plt.savefig('main_plot.jpg')
    plt.show()




if __name__ == '__main__':
    main()
