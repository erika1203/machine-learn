#!/usr/bin/env python3
# coding: utf-8

import numpy
from numpy import *   #科学计算包
import operator #运算符模块

def createDataSet():    #使用python导入数据
        group=array([[1.0,0.1],[1.0,1.0],[0,0],[0,0.1]])
        labels=['A','A','B','B']
        return group,labels

def classify0(input_set, data_set, labels, k):
        '''
        :param input_set: 输入向量
        :param data_set: 输入的样本数据集
        :param labels: 样本的标签向量，其个数与样本数据集的行数相同
        :param k: 选择近邻的个数
        :return: 输入向量所属的标签
        '''
        data_set_size=data_set.shape[0]    #shape[0]表示矩阵dataSet的行数，即坐标点个数
        diff_mat= tile(input_set, (data_set_size, 1)) - data_set   #将输入向量扩展为同样本集一样的结构，相减获得差值
        sq_diffmat=diff_mat**2    #矩阵中每一个元素都平方
        sq_distances=sq_diffmat.sum(axis=1)   #sum方法返回只有一行的矩阵，当axis=1，矩阵每行的列相加
                                            # 当axis=0时，每列的行相加
        distances=sq_distances**0.5  #矩阵中每一个元素开方，得到距离值
        sort_distances=distances.argsort() #按元素从小到大排序，返回下标
        labels_count={}
        for i in range(k):
            vote_label=labels[sort_distances[i]]
            print(sort_distances[i])
            labels_count[vote_label]=labels_count.get(vote_label,0)+1   #统计距离最近的labels的个数
        sorted__labels_count=sorted(labels_count.items(),key=operator.itemgetter(1),reverse=True)
        print(sq_diffmat,sq_distances,distances,sort_distances,labels_count)
        return sorted__labels_count[0][0]


