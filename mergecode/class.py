#!/home/ict2080ti/anaconda3/bin/python3
import argparse
import re
import os
from gensim.models.doc2vec import Doc2Vec
import torch
import numpy as np
import new_RNN

re1 = re.compile(r'%[a-z0-9]+')
re2 = re.compile(r'\*%[a-z0-9]+')
re3 = re.compile(r'\(%[a-z0-9]+\)')
re4 = re.compile(r'-?0[xX][0-9a-fA-F]+\(%[a-z0-9]+\)')
re5 = re.compile(r'\(%[a-z0-9]+,%[a-z0-9]+,-?([0-9]|\b0[xX][0-9a-fA-F]+\b)+\)')
re6 = re.compile(r'-?0[xX][0-9a-fA-F]+\(%[a-z0-9]+,%[a-z0-9]+,-?([0-9]|\b0[xX][0-9a-fA-F]+\b)+\)')
re7 = re.compile(r'\b0[xX][0-9a-fA-F]+\b')
re8 = re.compile(r'\$\b0[xX][0-9a-fA-F]+\b')

parser = argparse.ArgumentParser(description='Block Classification')
parser.add_argument('--path', type=str, required=True)
args = parser.parse_args()

#read log by filename
def readlog(filename):
    file = open(filename)
    trace = []
    trace_temp = []
    map = {}
    blocks = {}
    temp = []
    while 1:
        lines = file.readlines(100000)
        if not lines:
            break
        for line in lines:
            t_list = line.split()
            #t_list = re.split('[,\s]*', line)
            if line[0] == 'T':
                # print(line)
                if line[-3] != ']':
                    trace.append(line[21:-1])
                    map[trace[-1]] = t_list[1]
                    trace_temp.append(t_list[1])
                continue
            if line[0] == 'O' or line[0] == 'P':
                continue
            #temp.append(line[:-1].split())
            temp.append(re.split('[,\s]+', line[:-1]))
            if line[0] == "\n":
                temp = temp[:-1]
                blocks[temp[0][0][:-1]] = temp
                # if (temp[0][0][:-1] in trace_temp):
                #     blocks[temp[0][0][:-1]] = temp
                temp = []
                continue
    file.close()
    return trace, map, blocks

#preprocess and regularize
def process(blocks):
    for key in blocks:
        temp = blocks[key]
        for i in temp:
            del i[0]
            if len(i) == 2:
                if i[-1] == '':
                    # print(i)
                    del i[-1]
                    i.append('0')
                i.append('0')
            for j in i:
                if re6.match(j):
                    i[i.index(j)] = '6'
                    continue
                if re5.match(j):
                    i[i.index(j)] = '5'
                    continue
                if re4.match(j):
                    i[i.index(j)] = '4'
                    continue
                if re3.match(j):
                    i[i.index(j)] = '3'
                    continue
                if re2.match(j):
                    i[i.index(j)] = '2'
                    continue
                if re1.match(j):
                    i[i.index(j)] = '1'
                    continue
                if re7.match(j):
                    i[i.index(j)] = '7'
                    continue
                if re8.match(j):
                    i[i.index(j)] = '8'
                    continue
                if j == '#':
                    haha = i
                    temp[temp.index(i)] = haha[:i.index(j)]
    return blocks

#padding and turn to 40*300
def getVec(one_data):
    model = Doc2Vec.load("block2vecDM_1.model")
    temp_one = []
    if len(one_data) > 40:
        one_data = one_data[len(one_data) - 40:]
    for block in one_data:
        vec = model.infer_vector(block)
        temp_one.append(vec)
    if len(temp_one) < 40:
        while len(temp_one) < 40:
            temp_one.append(np.zeros((300,), dtype='float32'))
    return temp_one

#classify
def classify(one_data):
    config = new_RNN.Config()
    model = new_RNN.Model(config).to(config.device)
    model.load_state_dict(torch.load('new_RNN11.ckpt'))
    one_data = [one_data]
    one_data = torch.FloatTensor(one_data)
    res = model(one_data)
    predic = torch.max(res.data, 1)[1].cpu().numpy()
    return predic

#write to file
def writeToFile(filename, new_trace, final_blocks):
    with open(filename + "_re.log", 'w') as n:
    # with open("../d_re_neg/" + filename + "_re.log", 'w') as n:
        for i in range(len(new_trace)):
            for j in final_blocks[i]:
                n.write(' '.join(j) + ' ')
            n.write(new_trace[i] + '\n')

if __name__ == "__main__":
    # print(args.path)
    log_path = (args.path)
    #log_path = 'good2.log'
    trace, map, blocks = readlog(log_path)

    #temp_block是根据map选的blocks，这里应该可以做一些处理的
    if len(trace) > 40:
        trace = trace[len(trace) - 40:]
    res_map = {}
    for tb in trace:
        res_map[tb] = map[tb]
    map = res_map

    temp_block = {}
    for key in map:
        temp_block[map[key]] = blocks[map[key]]
    process(temp_block)

    res_blocks = []
    for i in trace:
        temp_i = []
        for ins in temp_block[map[i]]:
            temp_i.extend(ins)
        res_blocks.append(temp_i)
    #writeToFile(log_path, trace, res_blocks)

    vec = getVec(res_blocks)
    c = classify(vec)
    if c[0] == 0:
        print("neg!!!")
        if os.path.isfile(log_path):
            os.remove(log_path)
