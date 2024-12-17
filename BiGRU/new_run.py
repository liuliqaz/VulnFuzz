import time
import torch
import numpy as np
from new_train_eval import train, init_network
from importlib import import_module
import argparse
from tensorboardX import SummaryWriter
import new_utils

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集
    model_name = 'models.new_RNN'

    x = import_module(model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print("Loading data...")
    #经过这一步build_dataset，得到经过padding的句子list，每个词对应的是一个索引
    train_data, test_data, dev_data = new_utils.integrate_data()
    train_iter = new_utils.build_iterator(train_data, config)
    dev_iter = new_utils.build_iterator(dev_data, config)
    test_iter = new_utils.build_iterator(test_data, config)
    time_dif = new_utils.get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter,writer)
