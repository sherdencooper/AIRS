# %%
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
import os
import random
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
from Input_Cell_Attention.Scripts.cell import *
from torch.utils.tensorboard import SummaryWriter
from train_theta import fix_theta
from train_attention import Attention_model
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
from train_concept import generate_dataset,Concept_model
from train_theta import fix_theta
from train_attention import Attention_model
from train_cell import LSTM_Input_Cell
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# %%
def minmaxscaler_saliency(grad):
    min = np.amin(grad)
    max = np.amax(grad)    
    return (grad - min)/(max-min)

def concept_map(model,index,test_data):
    model.zero_grad()
    x = torch.unsqueeze(test_data[index], axis=0).to(device)
    output,theta = model(x,output_explain=True)
    grad = theta.detach().cpu().numpy()[0]
    return grad, np.abs(grad)

def theta_map(model):
    weight = model.fc3.weight[0].detach().cpu().numpy()
    return weight, np.abs(weight)

def attention_map(model,index,test_data):
    model.eval()
    x = torch.unsqueeze(test_data[index], axis=0).to(device)
    output,attention = model(x,output_explain=True)
    attention = attention.detach().cpu().numpy()
    return attention.squeeze(),np.abs(attention).squeeze()

def saliency_map(net, index, test_data):
    # with torch.backends.cudnn.flags(enabled=False):
    x = torch.unsqueeze(test_data[index], axis=0).to(device)
    net.zero_grad()
    feed_ = x.float()
    feed_.requires_grad = True
    feed_.retain_grad()
    h0 = torch.zeros(1, feed_.size(0), net.h_size).to(device)
    c0 = torch.zeros(1, feed_.size(0), net.h_size).to(device)
    out, (h_n, h_c) = net.lstm(feed_, (h0, c0))
    feed = net.fc1(out[:, -1, :])
    feed = F.sigmoid(feed)
    # feed = net.dropout1(feed)
    feed = net.fc2(feed)
    feed = F.sigmoid(feed)
    feed = net.fc3(feed)
    feed = F.sigmoid(feed)
    feed = net.fc4(feed)
    feed.backward()
    grad = feed_.grad.data.cpu().numpy().squeeze()
    abs_grad = np.abs(grad)
    return grad, abs_grad

def integrated_grad(net, index, test_data,steps,baseline):
    # with torch.backends.cudnn.flags(enabled=False):
    x = torch.unsqueeze(test_data[index], axis=0).to(device)
    net.zero_grad()
    feed_ = x.float()
    feed_.requires_grad = True
    if baseline is None:
        baseline = 0 * feed_
    else:
        baseline = baseline * feed_
    inputs = [baseline + (float(i) / steps) * (feed_ - baseline) for i in range(0, steps + 1)]
    grads = []
    for scaled_inputs in inputs:
        scaled_inputs.requires_grad_()
        scaled_inputs.retain_grad()
        h0 = torch.zeros(1, scaled_inputs.size(0), net.h_size).to(device)
        c0 = torch.zeros(1, scaled_inputs.size(0), net.h_size).to(device)
        out, (h_n, h_c) = net.lstm(scaled_inputs, (h0, c0))
        feed = net.fc1(out[:, -1, :])
        feed = F.sigmoid(feed)
        # feed = net.dropout1(feed)
        feed = net.fc2(feed)
        feed = F.sigmoid(feed)
        feed = net.fc3(feed)
        feed = F.sigmoid(feed)
        feed = net.fc4(feed)
        net.zero_grad()
        feed.backward(retain_graph=True)
        gradients = scaled_inputs.grad.data.cpu().numpy()[0]
        grads.append(gradients)
    grads = np.array(grads)
    # print(grads.shape)
    avg_grads = np.average(grads[:-1], axis=0)
    delta_X = (feed_ - baseline).detach().squeeze(0).cpu().numpy()
    integrated_grad = delta_X * avg_grads
    abs_grad = np.abs(integrated_grad)
    return integrated_grad, abs_grad

def random_baseline_integrated_gradients(net, index, test_data,steps,num_random_trials):
    all_intgrads = []
    for i in range(num_random_trials):
        grad = integrated_grad(net, index, test_data,steps,baseline = random.random())
        all_intgrads.append(grad)
    
    avg_intgrads = np.average(np.array(all_intgrads),axis=0)
    return avg_intgrads

def smoothgrad(net, index, test_data, n_samples = 25, stdev_spread=0.05):
    # with torch.backends.cudnn.flags(enabled=False):
    x = torch.unsqueeze(test_data[index], axis=0).to(device)
    net.zero_grad()
    feed_ = x.float()
    feed_.requires_grad = True
    total_grad = []
    stdev = stdev_spread * (torch.max(feed_) - torch.min(feed_))
    for i in range(n_samples):
        noise = torch.normal(torch.zeros(feed_.shape).to(device),stdev)
        feed_noise = feed_ + noise
        feed_noise.retain_grad()
        h0 = torch.zeros(1, feed_noise.size(0), net.h_size).to(device)
        c0 = torch.zeros(1, feed_noise.size(0), net.h_size).to(device)
        out, (h_n, h_c) = net.lstm(feed_noise, (h0, c0))
        feed = net.fc1(out[:, -1, :])
        feed = F.sigmoid(feed)
        # feed = net.dropout1(feed)
        feed = net.fc2(feed)
        feed = F.sigmoid(feed)
        feed = net.fc3(feed)
        feed = F.sigmoid(feed)
        feed = net.fc4(feed)
        feed.backward(retain_graph=True)
        grad = feed_noise.grad.data.cpu().numpy().squeeze()
        total_grad.append(grad)

    avg_intgrads = np.average(np.array(total_grad),axis=0)
    abs_grad = np.abs(avg_intgrads)
    return avg_intgrads, abs_grad

# %%
def write_summary(num, net, test_data, test_label, target_dir, method, grad_method,start,end):
    writer = open(target_dir,'w')
    for index in range(num):
        to_write = ""

        if grad_method == 'attention':
            grad, abs_grad = attention_map(net, index, test_data)
        if grad_method == 'theta':
            grad, abs_grad = theta_map(net)
        if grad_method == 'saliency':
            grad, abs_grad = saliency_map(net, index, test_data)
        if grad_method == 'concept':
            grad, abs_grad = concept_map(net, index, test_data)
        if grad_method == 'integrated':
            grad, abs_grad = integrated_grad(net, index, test_data,steps=5,baseline=None)
        if grad_method == 'smooth':
            grad, abs_grad = smoothgrad(net, index, test_data, n_samples=5, stdev_spread=0.05)


        if method == 'a_mean':
            if grad_method == 'saliency' or grad_method == 'smooth' or grad_method == 'integrated':
                abs_grad = np.mean(abs_grad, axis=1)
            scale = minmaxscaler_saliency(abs_grad)[start:end]
 
        max_loc = np.argmax(scale)+start
        min_loc = np.argmin(scale)+start
        to_write += str(max_loc)+";"+str(min_loc)+'\n'
        writer.writelines(to_write)
    writer.close()

# %%

if __name__ == '__main__':
    train_data,test_data,evaluate_data,train_label,test_label, evaluate_label = generate_dataset()
    device = torch.device("cuda")
    method = 'a_mean'
    num = 2000
    # to_test = ['attention','concept','theta','saliency','smooth','integrated']
    to_test = ['concept']  #for retraining
    for grad_method in to_test:
        target_dir = 'summary_explain/' + grad_method + '.txt'
        model_path = './model_save/'+grad_method+'.pth'
        if grad_method == 'saliency' or grad_method == 'smooth' or grad_method == 'integrated':
            model_path = './model_save/cell.pth'
        net = torch.load(model_path)
        start = 322
        end = 622
        write_summary(num,net,train_data,train_label,target_dir,method,grad_method,start,end)
        print(grad_method + ' Done.')