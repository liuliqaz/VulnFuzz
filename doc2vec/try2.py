
# -------------Let’s start implementing----------------------
# Import all the dependencies
import os
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

TaggedDocument = gensim.models.doc2vec.TaggedDocument

def readLog():
    x_train = []
    i = 1
    for filename in os.listdir("./"):
        if filename.endswith('.log'):
            file = open(filename)
            while 1:
                line = file.readline()
                if not line:
                    break
                list = line.split()
                wordlist = list[9:-1]
                tag = list[-1]
                # tag = str(i)
                # i += 1
                document = TaggedDocument(wordlist, tags=tag)
                x_train.append(document)
            file.close()
    return x_train

def train(x_train, size = 300):
    model = Doc2Vec(x_train, min_count=1, window=3, vector_size=size, sample=1e-3, nagative=5, workers=4)
    model.train(x_train, total_examples=model.corpus_count, epochs=10)
    return model

def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a,b)/(a_norm * b_norm)
    return cos

if __name__ == "__main__":
    # data = readLog()
    # print("read done")
    # model_dm = train(data)
    # print("train done")
    # model_dm.save("doc2vec2.model")
    #test = "mov 4 1 test 1 1 jl 7 0 mov 8 1 mov 1 4 movq 8 4 movq 8 4 movq 8 4 mov 8 1 mov 1 4 movq 8 4 mov 8 1 mov 1 4 mov 1 4 mov 1 4 mov 8 1 mov 1 4 movl 8 4 xchg 1 1 jmpq 7 0 mov 8 1 mov 1 4 lea 4 1 jmpq 7 0 lea 4 1 jmpq 7 0 0000004000a06ef5"

    model_dm = Doc2Vec.load("doc2vec2.model")


    step = 600

    test = "jmpq 7 0"
    list = test.split()
    vec = model_dm.infer_vector(doc_words=list, alpha=0.025, steps=step)
    #print(vec)

    test2 = "jne 7 0"
    list2 = test2.split()
    vec2 = model_dm.infer_vector(doc_words=list2, alpha=0.025, steps=step)
    #print(vec2)

    test3 = "sub 8 1"
    list3 = test3.split()
    vec3 = model_dm.infer_vector(doc_words=list3, alpha=0.025, steps=step)

    test4 = "add 1 1"
    list4 = test4.split()
    vec4 = model_dm.infer_vector(doc_words=list4, alpha=0.025, steps=step)


    print(cos_sim(vec, vec2))
    print(cos_sim(vec3,vec4))
    print(cos_sim(vec4, vec))

    # t1 = np.array([-0.4, 0.8, 0.5, -0.2, 0.3])
    # t2 = np.array([-0.5, 0.4, -0.2, 0.7, -0.1])
    # print(cos_sim(t1,t2))

    cc = "mov 4 1 add 1 1"
    dd = "mov 3 1 add 8 1 mov 4 1"
    ee = "lea 4 1 jmpq 7 0 mov 4 1"
    cc_list = cc.split()
    dd_list = dd.split()
    ee_list = ee.split()
    vc = model_dm.infer_vector(doc_words=cc_list, alpha=0.025, steps=step)
    vd = model_dm.infer_vector(doc_words=dd_list, alpha=0.025, steps=step)
    ve = model_dm.infer_vector(doc_words=ee_list, alpha=0.025, steps=step)

    print("c vs d", cos_sim(vc, vd))
    print("d vs e", cos_sim(vd, ve))
