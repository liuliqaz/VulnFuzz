import os
import sys
import numpy as np
import gensim

from gensim.models.doc2vec import Doc2Vec,LabeledSentence
from sklearn.model_selection  import train_test_split

TaggedDocument = gensim.models.doc2vec.TaggedDocument

def writeToFile(filename, things):
    with open(filename, 'w', encoding='utf-8') as n:
    # with open("../d_re_neg/" + filename + "_re.log", 'w') as n:
        for i in things:
            for j in i:
                n.write(' '.join(j) + ' ')
                n.write('\n')

def tt_get_dataset():
    sum = 0
    #读取数据
    os.chdir(".\\re_neg")
    pos_reviews = []
    for filename in os.listdir("./"):
        if filename.endswith('.log'):
            print(filename)
            file = open(filename)
            a = file.readlines()
            if (len(a)) > 100:
                sum += 100
            else:
                sum += len(a)
            temp = {}
            for i in a:
                i = i.split()
                temp[i[-1]] = i[9:-1]
            kk = []
            for key in temp:
                kk.append(temp[key])
            pos_reviews.append(kk)
            file.close()
    writeToFile("re_pos.txt", pos_reviews)
    print(sum/len(pos_reviews))

def kk_get_dataset():
    #读取数据
    os.chdir(".\\re_pos")
    pos_reviews = []
    temp = {}
    for filename in os.listdir("./"):
        if filename.endswith('.log'):
            print(filename)
            file = open(filename)
            a = file.readlines()
            for i in a:
                i = i.split()
                temp[i[-1]] = i[9:]
            file.close()
    kk = []
    for key in temp:
        kk.append(temp[key])
    pos_reviews.append(kk)
    writeToFile("re_pos1.txt", pos_reviews)

def get_dataset():
    #读取数据
    with open("re_pos.txt",'r', encoding='utf-8') as infile:
        pos_reviews = infile.readlines()
    with open("re_neg.txt",'r', encoding='utf-8') as infile:
        neg_reviews = infile.readlines()
    pos_reviews = list(set(pos_reviews))
    neg_reviews = list(set(neg_reviews))
    x_train = []
    for i in pos_reviews:
        x_train.append(i.split())
    for i in neg_reviews:
        x_train.append(i.split())

    def docuReview(doc):
        labelized = []
        for i, v in enumerate(doc):
            labelized.append(TaggedDocument(v, [str(i)]))
        return labelized

    x_train = docuReview(x_train)
    return x_train

def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)

def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a,b)/(a_norm * b_norm)
    return cos

if __name__ == "__main__":
    # tt_get_dataset()
    # kk_get_dataset()

    """
    size, epoch_num = 300, 10
    x_train = get_dataset()
    model_dm = Doc2Vec.load("block2vecDM_1.model")

    tt = ['mov', '4', '1', 'sub', '8', '1', 'mov', '4', '1', 'mov', '1', '4',
          'mov', '1', '4', 'mov', '1', '1', 'mov', '1', '4', 'lea', '4', '1',
          'mov', '4', '1', 'mov', '1', '3', 'lea', '4', '1', 'mov', '3', '1',
          'mov', '1', '3', 'mov', '8', '1', 'mov', '1', '3', 'mov', '3', '1',
          'add', '8', '1', 'mov', '1', '4', 'mov', '1', '4', 'mov', '4', '1',
          'add', '8', '1', 'mov', '1', '4', 'mov', '1', '4', 'jmpq', '7', '0',
          'lea', '4', '1', 'jmpq', '7', '0']
    print(tt)
    v1 = model_dm.infer_vector(tt)
    v2 = model_dm.infer_vector(tt)
    v3 = model_dm.docvecs['0']

    print(v1)
    print(v2)
    print(v3)

    print("c vs d", cos_sim(v1, v2))
    """

    """train with tag like '1' """
    size, epoch_num = 300, 10

    x_train = get_dataset()

    # 实例DM和DBOW模型
    model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
    model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

    # 使用所有的数据建立词典
    model_dm.build_vocab(x_train)
    model_dbow.build_vocab(x_train)

    # 进行多次重复训练，每一次都需要对训练数据重新打乱，以提高精度
    all_train_reviews = x_train
    for epoch in range(epoch_num):
        model_dm.train(all_train_reviews, total_examples=model_dm.corpus_count, epochs=model_dm.iter)
        model_dbow.train(all_train_reviews, total_examples=model_dbow.corpus_count, epochs=model_dbow.iter)

    model_dm.save("block2vecDM_2.model")
    model_dbow.save("block2vecDB_2.model")
    print("Model Saved")

    doc1 = x_train[0].words
    tag1 = x_train[0].tags
    print(doc1)
    print(tag1)

    v1 = model_dm.infer_vector(doc1)
    v2 = model_dm.infer_vector(doc1)
    v3 = model_dm.docvecs['0']
    print(cos_sim(v1, v2))
