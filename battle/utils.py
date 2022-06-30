from train_concept import generate_dataset
import numpy as np
import glob
import pandas as pd

def show_correlation():
    train_data,test_data,evaluate_data,train_label,test_label, evaluate_label = generate_dataset()
    average = []
    for i in range(8):
        a = []
        for j in range(64):
            train_data_tmp = train_data[:,j,i]
            a.append(np.correlate(train_data_tmp.flatten(), train_label.flatten())[0])
        average.append(a)
    average = np.array(average)
    average = np.mean(average, axis=0)
    print(average) 
    print(np.argsort(average))


def extract_reward():
    train_dir = './observation/train/'
    save_dir = './observation/reward.txt'
    obs_file = open(save_dir,'w')
    train_list = glob.glob(train_dir+'*.txt')
    for i in range(2000):
        trajetory = train_dir + str(i) + '.txt'
        data = pd.read_table(trajetory, sep=';', header=None)
        obs_file.write(str(data.values[-1:,0:1][0][0]) +'\n')

def select_abnormal():
    train_dir = './observation/train/'
    train_list = glob.glob(train_dir+'*.txt')
    i = 0
    for trajetory in train_list:
        data = pd.read_table(trajetory, sep=';', header=None)
        # print(data.values[:-1,:].shape)
        if data.values[:-1,:].shape[0] > 65:
            print(i)
        i+=1

def cal_win_rate():
    path = './observation/reward.txt'
    data = pd.read_table(path, sep=';', header=None).values
    num = 0
    for i in range(data.shape[0]):
        if data[i] >= 5000:
            num+=1
    print("The average winning rate is: ", num/data.shape[0])

cal_win_rate()