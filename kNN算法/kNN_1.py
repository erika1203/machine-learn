#!/usr/bin/env python3
# coding: utf-8

import numpy
from numpy import *   #科学计算包
import operator #运算符模块
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def init_data():
    '''
    从txt文档中读取数据组成矩阵
    '''
    f=open('datingTestSet.txt','r')
    content=f.readlines()
    rows=len(content)
    return_mat=ones((rows,3))   #构造一个矩阵，先用1填充，rows行3列
    class_label_vec=[]          #这是分类标签列表
    index=0
    for row in [value.split('\t') for value in content]:    #用空格符将文本中的数字分隔开
        return_mat[index,:]=row[0:3]    #依行取content前3列的特征数据放进矩阵中
        class_label_vec.append(int(row[-1]))    #取出分类标签放置在列表中
        index+=1
    f.close()
    return return_mat,class_label_vec   #返回特征数据矩阵，分类标签列表

def feature_scaling(dataset):
    '''
    数据归一化，因为数据中特征值之间差值太大，（此例中飞行里程数远高于其他特征值），会影响分类结果
    为了使结果更可靠，采用数据归一化处理，即使用公式
    newvalue=(oldvalue-min)/(max-min)
    将特征值转化为0到1之间的数值
    '''
    max_value=dataset.max(0)    #max(0)表示从列中找出最大值，组成一个1*3的矩阵
    min_value=dataset.min(0)    #同上，取列最小值
    diff_value=max_value-min_value  #由max-min的值组成的矩阵
    norm_dataset=ones(shape(dataset))   #构造一个与dataset同结构的矩阵
    m=dataset.shape[0]  #行数
    norm_dataset=dataset-tile(min_value,(m,1))  #用tile使min矩阵成为m*3的矩阵，从而能进行减法运算
    norm_dataset=norm_dataset/tile(diff_value,(m,1))    #同上，进行相除运算
    return norm_dataset,diff_value,min_value
    #除了归一化的矩阵之外，我们在后面还需要取值范围和最小值归一化测试数据

def make_plot():
    '''
    使用matplotlib画出散点图
    '''
    x,y=init_data()
    norm_mat,diff_value,min_value=feature_scaling(x)
    fig=plt.figure()    #绘制图表
    ax=fig.add_subplot(111) #创建1*1的图表矩阵，子图为序号1（行数，列数，索引）
    ax.scatter(x[:,1],x[:,2],15.0*asarray(y),15.0*asanyarray(y))    #取2，3列绘图，按分类列表y进行个性化标记
    plt.xlabel('playing video games percentage')  #x轴标签
    plt.ylabel('eating icecream per week') #y轴标签
    plt.show()  #显示图表

def classify0(input_set,data_set,labels,k):
    '''
    具体实现kNN算法
    '''
    data_set_size=data_set.shape[0]
    diff_mat=tile(input_set,(data_set_size,1))-data_set
    sq_diff_mat=diff_mat**2
    sq_distances=sum(sq_diff_mat,axis=1)
    distances=sq_distances**0.5
    sort_distances=argsort(distances)
    labels_count={}
    for i in range(k):
        vote_label=labels[sort_distances[i]]
        labels_count[vote_label]=labels_count.get(vote_label,0)+1
    sort_lables_count=sorted(labels_count.items(),key=operator.itemgetter(1),reverse=True)
    return sort_lables_count[0][0]

def classify_main():
    '''
    约会对象分类实例
    '''
    result_list=['not at all','in small doses','in large doses']   #各分类类别
    fly_miles=float(input('enter your number of flying miles per year:'))    #输入特征数据1
    game_time=float(input('enter percentage of time you spent on playing video games per week:')) #输入特征数据2
    eat_icecream=float(input('enter hwo many icecreams you eat per year:'))      #输入特征数据3
    data_set,labels=init_data()         #获得训练集
    norm_mat,diff_value,min_value=feature_scaling(data_set)      #得到归一化矩阵和范围参数
    input_set=asarray([fly_miles,game_time,eat_icecream])        #形成输入数据集
    classify_result=classify0((input_set-min_value)/diff_value,data_set,labels,3)  #对输入数据集进行归一化处理后进行kNN预测
    print('You will probably like this kind of person:',result_list[classify_result-1])    #获得分类结果

def dating_class_test():
    '''
    一般用90%数据来训练，10%数据作测试，数据挑选随机
    '''
    test_ratio=0.1    #测试率
    dating_data_mat,dating_labels=init_data()      #获得训练集
    norm_mat,diff_value,min_value=feature_scaling(dating_data_mat)  #获得归一化矩阵和范围参数
    m=norm_mat.shape[0]              #归一化矩阵的行数（及总样本数据集的个数）
    num_test=int(m*test_ratio)  #测试集的个数
    error_count=0.0  #错误个数
    for i in range(num_test):
        #测试数据
        classify_result=classify0(norm_mat[i,:],norm_mat[num_test:m,:],dating_labels[num_test:m],4)
        #测试集逐一被当成输入集，样本数据集变为测试集以外剩下的数据
        print('The classifier came back with :%d,the real answer is :%d' % (classify_result,dating_labels[i]))
        if classify_result != dating_labels[i]:
            error_count+=1
    right_ratio=1-error_count/float(num_test)   #计算正确率
    print('The total right rate is :%f %%' % (right_ratio*100))



if __name__=='__main__':
    make_plot()
    dating_class_test()
    classify_main()
    









