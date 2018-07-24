import re
from numpy import *

def load_data():
    f=open('adults.txt','r')
    data_list=[]
    class_list=[]
    for line in f.readlines():
        data=re.split(r',',line.strip())
        data=[i.strip(' ') for i in data]
        if '?' not in data:
            if data[-1]=='>50K':
                class_list.append(1)
            else:
                class_list.append(0)
            del(data[-1])
            data_list.append(data)
    print(data_list)
    print(class_list)
    return data_list,class_list

def vocab_set(data_list):
    ret_vec=set([])
    for vec in data_list:
        vec_set=set(vec)
        ret_vec=ret_vec | vec_set
    return list(ret_vec)


def train_data(train_list,class_list):
    train_num=len(train_list)
    tag_num=len(train_list[0])
    p1_rate=class_list.count(1)/float(train_num)
    p0_vec=ones((tag_num,1))
    p1_vec=ones((tag_num,1))


