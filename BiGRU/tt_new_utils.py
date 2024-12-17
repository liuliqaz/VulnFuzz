# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
from sklearn.model_selection  import train_test_split
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize
from random import sample
import random
from importlib import import_module

#读取所有数据
def tt_get_dataset():
    #读取数据
    pos_reviews = []
    count = 0
    for filename in os.listdir("./re_pos/"):
        if filename.endswith('.log'):
            print(filename)
            file = open('./re_pos/' + filename)
            a = file.readlines()
            temp = []
            for i in a:
                i = i.split()
                temp.append(i[9:-1])
            pos_reviews.append(temp)
            file.close()
            count += 1
        if count == 10:
            break
    print("--------------------------------------------------------------------------------------")
    neg_reviews = []
    count = 0
    for filename in os.listdir("./re_neg/"):
        if filename.endswith('.log'):
            print(filename)
            file = open('./re_neg/' + filename)
            a = file.readlines()
            temp = []
            for i in a:
                i = i.split()
                temp.append(i[9:-1])
            neg_reviews.append(temp)
            file.close()
            count += 1
        if count == 10:
            break
    return pos_reviews, neg_reviews

#将基本块序列转为向量1753*40*300：1753条数据，每条40个基本块，每个基本块表示为300维向量
def getVec(reviews):
    model = Doc2Vec.load("block2vecDM_1.model")
    a = []
    for one_data in reviews:
        temp_one = []
        if len(one_data) > 40:
            one_data = one_data[len(one_data) - 40:]
        for block in one_data:
            vec = model.infer_vector(block)
            temp_one.append(vec)
        if len(temp_one) < 40:
            while len(temp_one) < 40:
                # temp_one.append([0 for i in range(300)])
                temp_one.append(np.zeros((300,), dtype='float32'))

        print(len(temp_one))
        a.append(temp_one)
    return a

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.FloatTensor([_[0] for _ in datas])
        x.requires_grad = True
        #x.is_leaf = True
        x.to(self.device)
        y = torch.LongTensor([_[1] for _ in datas])
        y.requires_grad = True
        #y.is_leaf = True
        y.to(self.device)
        return x, y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, 2, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

#读取所有的log文件，然后向量化，然后得到y
#完成padding工作以及划分训练集和测试集合x_train, x_test
#最后得到的应该是[([],0),([],1),...]
def integrate_data():
    pos_reviews, neg_reviews = tt_get_dataset()
    pos_reviews = getVec(pos_reviews)
    neg_reviews = getVec(neg_reviews)
    x = pos_reviews + neg_reviews
    #y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews)))).tolist()
    y = [1 for i in range(len(pos_reviews))] + [0 for i in range(len(neg_reviews))]
    datas = []
    for i in range(len(x)):
        datas.append((x[i], y[i]))
    random.shuffle(datas)
    train_data = datas[:12]
    test_data = datas[12:16]
    dev_data = datas[16:]

    return train_data, test_data, dev_data

if __name__ == "__main__":
    dataset = 'THUCNews'  # 数据集
    model_name = 'models.new_RNN'

    x = import_module(model_name)
    config = x.Config(dataset)

    start_time = time.time()
    print("Loading data...")
    # 经过这一步build_dataset，得到经过padding的句子list，每个词对应的是一个索引
    train_data, test_data, dev_data = integrate_data()
    print(len(train_data))
    print(len(test_data))
    print(len(dev_data))

    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    for i, (trains, labels) in enumerate(train_iter):
        a =trains
        print(i, labels)

