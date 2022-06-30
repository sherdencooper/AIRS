from train_concept import generate_dataset,padding
import numpy as np
import glob
import pandas as pd

def show_correlation():
    train_data,test_data,evaluate_data,train_label,test_label, evaluate_label = generate_dataset()
    average = []
    for i in range(50):
        a = []
        for j in range(1000):
            train_data_tmp = train_data[:,j,i]
            a.append(np.correlate(train_data_tmp.flatten(), train_label.flatten())[0])
        average.append(a)
    average = np.array(average)
    average = np.mean(average, axis=0)
    print(average) 
    print(np.argsort(average))
    window = []
    for i in range(700):
        sum = 0
        for j in range(300):
            sum+=average[i+j]
        window.append(sum)
    window = np.array(window)
    print(np.argsort(window))


def extract_reward():
    train_dir = './observation/train/'
    save_dir = './observation/reward.txt'
    obs_file = open(save_dir,'w')
    train_list = glob.glob(train_dir+'*.txt')
    for i in range(2000):
        trajetory = train_dir + str(i) + '.txt'
        data = pd.read_table(trajetory, sep=';', header=None)
        obs_file.write(str(data.values[-1:,0:1][0][0]) +'\n')


show_correlation()