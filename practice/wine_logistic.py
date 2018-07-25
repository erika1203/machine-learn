import re
from numpy import *

def load_data():
    f=open('wine.txt','r')
    data_list=[];label_list=[]
    for line in f.readlines():
        clean_line=re.split(',',line.strip())
        data=[float(i) for i in clean_line]
        if data[0]==1:
            label_list.append(1)
        else:
            label_list.append(0)
        del(data[0])
        data_list.append(data)
    return data_list,label_list

def sigmoid(x):
    sigm=1.0/(1+exp(-x))
    # print('The value of sigmoid is',sigm)
    return sigm

def grad_ascent(data_list,label_list,num_iter=100):
    data_mat=array(data_list)
    label_mat=array(label_list)
    m,n=shape(data_mat)
    weights=ones(n)
    for j in range(num_iter):
        data_index = list(range(m))
        for i in range(m):
            alpha=4/(1.0+i+j)+0.01
            rand_index=int(random.uniform(0,len(data_index)))
            h=sigmoid(sum(data_mat[rand_index]*weights))
            error=label_mat[rand_index]-h
            weights=weights+alpha*error*data_mat[rand_index]
            del(data_index[rand_index])
    return weights

def classify(input_data,weights):
    sigm=sigmoid(sum(input_data*weights))
    if sigm>0.5:
        return 1
    else:
        return 0

def train_test(test_rate=0.15,iter_num=50):
    err_rates=0.0
    for i in range(iter_num):
        data_list, label_list = load_data()
        data_num = len(data_list)
        test_num = int(data_num * test_rate)
        test_set = [];test_labels = []
        train_set = data_list;train_labels=label_list
        for i in range(test_num):
            rand_index=int(random.uniform(0,len(train_set)))
            test_set.append(train_set[rand_index])
            test_labels.append(train_labels[rand_index])
            del(train_set[rand_index])
            del(train_labels[rand_index])
        err_count=0.0
        weights=grad_ascent(train_set,train_labels)
        for i in range(test_num):
            result=classify(test_set[i],weights)
            if result != test_labels[i]:
                err_count+=1
        err_rates+=err_count/test_num
    avg_err=err_rates/iter_num
    print('The average error rate for (test rate: %.2f), (iter times:%d) is %.2f' % (test_rate,iter_num,avg_err))
    return avg_err

def commit():
    test_rates=[0.15,0.3,0.5]
    iter_nums=[30,50,100]
    for i in test_rates:
        for j in iter_nums:
            train_test(i,j)




