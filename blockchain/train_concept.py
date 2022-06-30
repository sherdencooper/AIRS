from cgi import test
import seaborn as sns
from numpy.random import randn
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import glob
import pandas as pd
import random
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
# from Input_Cell_Attention.Scripts.cell import *
from torch.utils.tensorboard import SummaryWriter
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def batch_jacobian(outputs, inputs, create_graph=False):
    """Computes the jacobian of outputs with respect to inputs

    :param outputs: tensor for the output of some function
    :param inputs: tensor for the input of some function (probably a vector)
    :param create_graph: set True for the resulting jacobian to be differentible
    :returns: a tensor of size (outputs.size() + inputs.size()) containing the
        jacobian of outputs with respect to inputs
    """
    jac = outputs.new_zeros(outputs.size() + inputs.size()
                            ).view((-1,) + inputs.size())
    for i, out in enumerate(outputs.view(-1)):
        col_i = torch.autograd.grad(out, inputs, retain_graph=True,
                              create_graph=create_graph, allow_unused=True)[0]
        if col_i is None:
            # this element of output doesn't depend on the inputs, so leave gradient 0
            continue
        else:
            jac[i] = col_i

    if create_graph:
        jac.requires_grad_()

    return jac.view(outputs.size() + inputs.size())

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def minmaxscaler(train_label, test_label, evaluate_label):
    min = np.amin(train_label)
    max = np.amax(train_label)    
    return (train_label - min)/(max-min), (test_label - min)/(max-min), (evaluate_label - min)/(max-min)

def padding(x, reward, initial_state = np.array([0.0,0.0,0.0])):
    if reward[-1]!=0:
        return x
    pointer = 0
    for i in range(x.shape[0],0,-1):
        pointer = i-1
        if (reward[i-1]!=0):
            break

    for i in range(x.shape[0], pointer+1, -1):
        x[i-1] = initial_state

    return x

def padding_(x, initial_state = np.array([0.0,0.0,0.0])):
    pointer = 0
    for i in range(x.shape[0],0,-1):
        pointer = i-1
        if (x[pointer] == initial_state).all():
            break

    for i in range(x.shape[0], pointer, -1):
        x[i-1] = initial_state

    return x

class Concept_model(nn.Module):
    def __init__(self, h_size, r=20, d_a=50):
        super(Concept_model, self).__init__()
        self.h_size = h_size
        # self.lstm = LSTMWithInputCellAttention(7, h_size, 20,50)
        self.lstm_t = nn.LSTM(7, h_size, 2, batch_first=True)
        self.lstm_h = nn.LSTM(7, h_size, 2, batch_first=True)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, 1)
        self.fc3 = nn.Linear(15, 1)
        self.emd1 = nn.Embedding(22,3)
        self.emd2 = nn.Embedding(22,3)
        self.emd3 = nn.Embedding(3,1)
        # self.fc4 = nn.Linear(194,256)
        # self.fc5 = nn.Linear(256,15)
        self.fc6 = nn.Linear(8, 4)
        self.fc7 = nn.Linear(4, 1)
    def forward(self, x, cal_reg=False, output_explain=False):
        input1 = self.emd1(x[:,:,0])
        input2 = self.emd2(x[:,:,1])
        input3 = self.emd3(x[:,:,2])
        temp = torch.cat((input1,input2), axis=2)
        x = torch.cat((temp, input3), axis=2)
        x.requires_grad_()
        x.retain_grad()

        h0 = torch.zeros(2, x.size(0), self.h_size).cuda()
        c0 = torch.zeros(2, x.size(0), self.h_size).cuda()
        theta, (h_n, h_c) = self.lstm_t(x, (h0, c0))
        theta = torch.reshape(theta,(-1,self.h_size))
        theta = self.fc1(theta)
        theta = self.fc2(theta)
        theta = torch.reshape(theta,(-1,15))


        def h_functions(x):
            h1 = torch.zeros(2, x.size(0), self.h_size).cuda()
            c1 = torch.zeros(2, x.size(0), self.h_size).cuda()
            h, (h_n, h_c) = self.lstm_h(x, (h1, c1))
            h = torch.reshape(h,(-1,self.h_size))
            h = self.fc6(h)
            h = self.fc7(h)
            h = torch.reshape(h,(-1,15))
            return h
        h = h_functions(x)
        h.requires_grad_()
        h.retain_grad()

        pred = theta.mul(h)
        pred = self.fc3(pred)
        if cal_reg:
            grad_f = torch.autograd.grad(pred.sum(), x, retain_graph=True)[0].data
            jacob_h = torch.autograd.functional.jacobian(h_functions,x)
            theta_times_jacob_h = theta.matmul(jacob_h.T).T
            reg = torch.sum((grad_f.cuda() - theta_times_jacob_h.cuda()) ** 2)
            return pred, reg
        elif output_explain:
            return pred, theta
        else:
            return pred

def generate_dataset(PADDING_DATASET=True,NORMALIZATION_OUTPUT=True,DRAW_DISTRIBUTION=False):

    BIG_DATASET = True
    PADDING_DATASET = False
    NORMALIZATION_OUTPUT = True
    DRAW_DISTRIBUTION = False
    USE_BEST_MODEL = True

    train_dir = './episode_15/train/'
    test_dir = './episode_15/test/'
    evaluate_dir = './episode_15/validate/'
    train_log_dir = './torch_log/big/train/' + TIMESTAMP
    test_log_dir = './torch_log/big/test/' + TIMESTAMP
    evaluate_log_dir = './torch_log/big/validate/' + TIMESTAMP
    net_log_dir = './torch_log/big/net/' + TIMESTAMP
    train_num = 10000
    test_num = 2000
    evaluate_num = 2000

    train_list = glob.glob(train_dir+'*.txt')
    test_list = glob.glob(test_dir+'*.txt')
    evaluate_list = glob.glob(evaluate_dir+'*.txt')
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    evaluate_data = []
    evaluate_label = []


    for trajetory in train_list:
        data = pd.read_table(trajetory, sep=';', header=None)
        reward = data[3].values
        df = data.drop([3], axis=1)
        if PADDING_DATASET:
            train_data.append(padding(df.values[:-1,:],reward[:-1]))
        else:
            train_data.append(df.values[:-1,:])
        train_label.append(df.values[-1:,0:1])

    for trajetory in test_list:
        data = pd.read_table(trajetory, sep=';', header=None)
        reward = data[3].values
        df = data.drop([3], axis=1)
        if PADDING_DATASET:
            test_data.append(padding(df.values[:-1,:],reward[:-1]))
        else:
            test_data.append(df.values[:-1,:])
        test_label.append(df.values[-1:,0:1])

    for trajetory in evaluate_list:
        data = pd.read_table(trajetory, sep=';', header=None)
        reward = data[3].values
        df = data.drop([3], axis=1)
        if PADDING_DATASET:
            evaluate_data.append(padding(df.values[:-1,:],reward[:-1]))
        else:
            evaluate_data.append(df.values[:-1,:])
        evaluate_label.append(df.values[-1:,0:1])

    train_data = np.array(train_data, dtype=np.int32)
    train_label = np.array(train_label, dtype=np.float32)
    test_data = np.array(test_data, dtype=np.int32)
    test_label = np.array(test_label, dtype=np.float32)
    evaluate_data = np.array(evaluate_data, dtype=np.int32)
    evaluate_label = np.array(evaluate_label, dtype=np.float32)

    if NORMALIZATION_OUTPUT:
        train_label, test_label, evaluate_label, = minmaxscaler(train_label, test_label, evaluate_label)

    # train_data = np.expand_dims(train_data, axis = 1)
    # test_data = np.expand_dims(test_data, axis = 1)
    # evaluate_data = np.expand_dims(evaluate_data, axis = 1)
    train_label = np.squeeze(train_label, axis = 2)
    test_label = np.squeeze(test_label, axis = 2)
    evaluate_label = np.squeeze(evaluate_label, axis = 2)

    if DRAW_DISTRIBUTION:

        plt.hist(train_label,bins=25,alpha=0.4,label='train')
        plt.hist(test_label,bins=25,alpha=0.8,label='test')
        plt.hist(evaluate_label,bins=25,alpha=0.8,label='evaluate')
        plt.legend(labels=['train','test','evaluate'],loc='best')
        plt.xlabel('Relative incentives')
        plt.ylabel('Number')
        plt.title("Label Distribution")
        plt.show()

    train_data = torch.from_numpy(train_data)
    test_data = torch.from_numpy(test_data)
    evaluate_data = torch.from_numpy(evaluate_data)
    train_label = torch.from_numpy(train_label)
    test_label = torch.from_numpy(test_label)
    evaluate_label = torch.from_numpy(evaluate_label)

    return train_data,test_data,evaluate_data,train_label,test_label, evaluate_label





def train(model, device, train_data, train_label, test_data, test_label, evaluate_data, evaluate_label, optimizer, epochs, scheduler,train_log_dir,test_log_dir,evaluate_log_dir,USE_BEST_MODEL=True,train_num=2000,test_num=100,evaluate_num=100,save_dir='./model_save/concept.pth'):
    batch_size = 50
    iteration = train_num // batch_size
    k = 1000
    loss = nn.MSELoss()
    loss_ = nn.MSELoss(reduce=False)
    label_mean = train_label.mean().to(device)
    history_best_loss = 10000
    for epoch in range(1,epochs+1):
        pre_index = 0
        train_loss = 0.0
        for step in range(iteration):
            # print(step)
            model.train()
            batch_x = train_data[pre_index : pre_index+batch_size].to(device)
            batch_y = train_label[pre_index : pre_index+batch_size].to(device)
            
            # output, reg = model(batch_x,cal_reg=True)
            # batch_loss = loss(output, batch_y) + 1e-6 * reg
            output= model(batch_x)
            batch_loss = loss(output, batch_y)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss
            pre_index += batch_size

        train_loss /= iteration # average loss
        evaluate_loss = evaluate(model, device, evaluate_data, evaluate_label, loss, test_num)

        with SummaryWriter(train_log_dir) as writer:
            writer.add_scalar('loss', train_loss, epoch)
        with SummaryWriter(evaluate_log_dir) as writer:
            writer.add_scalar('loss', evaluate_loss, epoch)

        line = "epoch: %d/%d, train_loss: %.4f, evaluate_loss: %.4f\n" % (
                                epoch, epochs, train_loss, evaluate_loss)
        print(line)
        scheduler.step()
        if USE_BEST_MODEL:
            if evaluate_loss <= history_best_loss:
                torch.save(model, save_dir)
                history_best_loss = evaluate_loss
            

    print("Training Finished. Now Testing")
    print("========================================================")
    if USE_BEST_MODEL:
        model = torch.load(save_dir)
    test_loss = evaluate(model, device, test_data, test_label, loss, test_num)
    line = "test_loss: %.4f" % (test_loss)
    with SummaryWriter(test_log_dir) as writer:
        writer.add_scalar('test_loss', test_loss)
    print(line)


def evaluate(model, device, test_data, test_label, loss, test_num):
    # model.eval()
    batch_size = 50
    iteration = test_num // batch_size
    test_loss = 0.0
    test_pre_index = 0
    with torch.no_grad():
        for step in range(iteration):
            test_batch_x = test_data[test_pre_index: test_pre_index+batch_size].to(device)
            test_batch_y = test_label[test_pre_index: test_pre_index+batch_size].to(device)
            output = model(test_batch_x)
            batch_loss = loss(output, test_batch_y)
            test_loss += batch_loss
            test_pre_index += batch_size

            if step == iteration - 1:
                test_loss /= iteration

    return test_loss



if __name__ == '__main__':
    setup_seed(2021)
    train_data,test_data,evaluate_data,train_label,test_label, evaluate_label = generate_dataset()
    USE_BEST_MODEL = True
    train_num = 10000
    test_num = 2000
    evaluate_num = 2000
    hiddne_size = 8
    epsilon = 1e-4
    gamma = 0.2
    decay_step = 50
    save_dir = './model_store/concept.pth'
    train_log_dir = './torch_log/train/' + TIMESTAMP
    test_log_dir = './torch_log/test/' + TIMESTAMP
    evaluate_log_dir = './torch_log/validate/' + TIMESTAMP
    net_log_dir = './torch_log/net/' + TIMESTAMP
    device = torch.device("cuda")
    net = Concept_model(h_size=hiddne_size).to(device) 
    dummy_input = torch.rand(15,15,3).long().to(device)
    # with SummaryWriter(net_log_dir) as w:
    #     w.add_graph(net, (dummy_input,))
    # optimizer = optim.Adadelta(net.parameters(), lr=1e-2)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=decay_step, gamma=gamma)
    epochs = 100
    train(net, device, train_data, train_label, test_data, test_label, evaluate_data, evaluate_label, optimizer, epochs, scheduler, train_log_dir,test_log_dir,evaluate_log_dir,USE_BEST_MODEL,train_num,test_num,evaluate_num,save_dir)
