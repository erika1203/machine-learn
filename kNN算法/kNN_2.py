#!/usr/bin/env python3
# coding:utf-8
#手写识别系统例子，手写数字0-9转化为2000个32*32位的01矩阵，存储在trainingDigits中

from numpy import *
from os import listdir
import operator

def img2vector(filename):
    #将32*32的矩阵转化为1*1024的向量返回
    vec=zeros((1,1024))  #构造一个1*1024的全0数组（向量）
    f=open(filename)    #打开一个32*32的矩阵文件
    for i in range(32):
        line=f.readline()#读取每行
        for j in range(32):
            vec[0,32*i+j]=int(line[j])
            #读取行中每列的值，放在向量对应位置中
            #32*i+j是一个将矩阵铺成向量对应位置的计算公式
    return vec

def classify0(input_set,data_set,labels,k):
    #kNN实现算法
    data_set_size=data_set.shape[0]
    diff_mat=tile(input_set,(data_set_size,1))-data_set
    sq_diff_mat=diff_mat**2
    sq_distances=sq_diff_mat.sum(axis=1)
    distances=sq_distances**0.5
    sort_ditances=distances.argsort()
    labels_count={}
    for i in range(k):
        vote_label=labels[sort_ditances[i]]
        labels_count[vote_label]=labels_count.get(vote_label,0)+1
    sort_labels_count=sorted(labels_count.items(),key=operator.itemgetter(1),reverse=True)
    return sort_labels_count[0][0]

def handwriting_test():
    hw_labels=[]
    file_list=listdir('trainingDigits') #listdir函数可以列出指定目录内的文件名
    print(file_list)
    m=len(file_list)
    data_mat=zeros((m,1024)) #初始化m*1024的全1矩阵
    for i in range(m):
        file_name=file_list[i]
        file_num=int((file_name.split('.')[0]).split('_')[0])   #从文件名里获知文件中存储的数字
        hw_labels.append(file_num)
        data_mat[i,:]=img2vector('trainingDigits/%s' % file_name)
         #读取文件夹内各文件，转为向量返回，放置在数据集矩阵中的各行上

    test_file_list=listdir('testDigits')    #另一个文件夹中用于测试的文件
    error_count=0.0
    n=len(test_file_list)
    for j in range(n):
        file_name=test_file_list[j]
        file_num=int((file_name.split('.')[0]).split('_')[0])   #存储的数字
        vec_test=img2vector('testDigits/%s' % file_name)
        classify_result=classify0(vec_test,data_mat,hw_labels,3)#测试test数据集
        print('\nThe classigyier came back with: %d,the real answer is %d' % (classify_result,file_num))
        if classify_result != file_num:
            error_count+=1          #计算错误个数
    right_ratio=1-error_count/float(n)
    print('\nThe total right rate is : %f %%' % (right_ratio*100)) #输出正确率

if __name__ == '__main__':
    print('The digits recognition begins...')
    handwriting_test()








