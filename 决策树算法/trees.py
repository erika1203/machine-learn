#!/usr/bin/env python3
# coding:utf-8

from math import log
import operator
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd

def calcShannonEnt(dataset):
    #计算香农熵
    num_entries=len(dataset)    #数据集中的实例个数
    label_counts={}             #创建字典以计算不同类别的个数
    for feat_vec in dataset:
        current_label=feat_vec[-1]  #最后一列的值（即类别）
        if current_label not in label_counts.keys():
            label_counts[current_label]=0   #加入一个新发现的类别
        label_counts[current_label]+=1  #计算各类别个数
    shanno_ent=0.0              #香农熵
    for key in label_counts:
        prob=float(label_counts[key]/num_entries)   #各类别出现在实例中的概率
        shanno_ent-=prob*log(prob,2)#按公式求香农熵
    return shanno_ent

def createDataset():
    #创建数据集
    dataset=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels=['no surfacing','flippers']
    return dataset,labels

def splitDataset(dataset,axis,value):
    '''
    划分数据集，按某特征某值划出决策树的一个节点
    :param dataset: 待划分的数据集
    :param axis: 划分数据值的特征下标
    :param value: 特征的数值
    :return: 划分后的数据集
    '''
    ret_dataset=[]  #返回的数据集
    for feat_vec in dataset:    #对原数据集中的各个列表元素
        if feat_vec[axis]==value:   #找出符合条件（即其特征值等于指定值）的元素
            reduced_featvec=feat_vec[:axis]     #返回除了此条件包含特征值以外的列表内容
            reduced_featvec.extend(feat_vec[axis+1:])   #注意区分列表的extent()和append()方法
            ret_dataset.append(reduced_featvec)     #得到符合某条件的数据集合（每条数据中除去作为条件的特征项）
    return ret_dataset

def chooseBestfeature2split(dataset):
    #找到最好的划分特征
    num_features=len(dataset[0])-1  #计算特征项的个数，数据集最后一列是分类类别
    base_entropy=calcShannonEnt(dataset)    #计算初始香农熵
    best_infogain=0.0   #最大信息增益
    best_feature=-1     #最好划分特征
    for i in range(num_features):   #按逐个特征项计算其作为划分标准的信息增益
        feat_list=[vec[i] for vec in dataset]   #取得某个特征项中不同的值放入列表中
        unique_vals=set(feat_list)  #去重
        new_entropy=0.0
        for value in unique_vals:   #按某个特征项的不同值划分数据集
            sub_dataset=splitDataset(dataset,i,value)   #原数据集中符合某个特征值的元素列表
            prob=len(sub_dataset)/float(len(dataset))   #该子集在总数据集中出现的概率
            new_entropy+=prob*calcShannonEnt(sub_dataset)   #计算新香农熵
        infogain=base_entropy-new_entropy       #计算信息增益
        print('第%d个特征的增益为：%.3f' % (i,infogain))
        if infogain > best_infogain:    #找到最大信息增益和对应的特征项下标
            best_infogain=infogain
            best_feature=i
    return best_feature

def majorityCnt(class_list):
    #计算包含元素最多的类别
    class_count={}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote]=0
        class_count[vote]+=1
    sort_class_count=sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)
    return sort_class_count[0][0]

def createTree(dataset,labels):
    #递归构建决策树，创建的决策树是一个层层嵌套的字典
    #区分这里的labels和class，labels是各特征项的名称，class是分类类别

    class_list=[vec[-1] for vec in dataset] #将数据集中各项元素的类别值放到列表中
    if class_list.count(class_list[0])==len(class_list):
        #若列表中所有类别相同，则表明按照某特征项已经可以确定类别了，停止继续划分类别
        return class_list[0]
    if len(dataset[0])==1:
        #若已经遍历完所有特征项，数据集中只剩下类别那一列，则返回出现次数最多的类别
        return majorityCnt(class_list)
    best_feat=chooseBestfeature2split(dataset)  #最好划分特征的下标
    best_feat_label=labels[best_feat]       #最好划分特征下标对应的特征名称
    #由于原labels已经被破坏，重新得到一个按最好特征排序的labels
    my_tree={best_feat_label:{}}    #()表示按此特征能够划分的内容，是下一级递归的返回值
    del(labels[best_feat])    #在特征名称列表中去掉已经使用过的特征项
    feat_values=[vec[best_feat] for vec in dataset] #取得所用特征项中的各个特征值放入列表中
    unique_values=set(feat_values)  #去重
    for value in unique_values:     #对特征值列表中的各个值构建决策树分支
        sub_labels=labels[:]        #去掉已用特征项的特征名称列表，用作下一次递归
        sub_dataset=splitDataset(dataset,best_feat,value)   #按某个特征值划分数据集
        my_tree[best_feat_label][value]=createTree(sub_dataset,sub_labels) #递归
    return my_tree

def getNumLeafs(my_tree):
    #采用递归方法获取决策树叶子节点的数目
    num_leafs=0     #初始化叶子数
    first_str=next(iter(my_tree))
    #获取类型为字典的决策树中的特征名称字段
    second_dict=my_tree[first_str]
    #获取该特征对应的分类结果
    for key in second_dict.keys():
        if type(second_dict[key]).__name__=='dict':
            num_leafs+=getNumLeafs(second_dict[key])
            #若分类结果为字典，说明其不是叶子节点，进行递归
        else:
            num_leafs+=1
            #若分类结果不是字典，则说明已得到分类结果，即是叶子节点
    return num_leafs

def getTreeDepth(my_tree):
    #采用递归方法获取决策树层数
    max_depth=0     #用max_depth固定层数，使不受递归影响
    first_str=next(iter(my_tree))
    second_dict=my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__=='dict':
            this_depth=1+getTreeDepth(second_dict[key])
            #当前已有一层，若分类结果为字典则继续遍历下一层，若不是则到达最后一层
        else:
            this_depth=1
        if this_depth > max_depth:
            max_depth=this_depth    #更新层数
    return max_depth

def plotNode(node_txt,center_pt,parent_pt,node_type):
    '''
    绘制决策树节点
    :param node_txt: 节点名
    :param center_pt: 文本位置
    :param parent_pt: 标注的箭头位置
    :param node_type: 节点格式
    :return:
    '''
    arrow_args=dict(arrowstyle='<-')        #箭头格式
    font=FontProperties(fname=r'/Library/Fonts/Microsoft/Fangsong.ttf',size=14)     #设置中文字体
    createPlot.ax1.annotate(node_txt,xy=parent_pt,xycoords='axes fraction',
                            xytext=center_pt,textcoords='axes fraction',
                            va='center',ha='center',bbox=node_type,arrowprops=arrow_args,FontProperties=font)
    #绘制节点

def plotMidtext(cntr_pt,parent_pt,txt_string):
    '''
    标注有向边的属性值
    :param cntr_pt: 计算标注位置
    :param parent_pt: 计算标注位置
    :param txt_string: 标注的内容
    :return:
    '''
    x_mid=(parent_pt[0]-cntr_pt[0])/2.0 + cntr_pt[0]
    y_mid=(parent_pt[1]-cntr_pt[1])/2.0 + cntr_pt[1]
    createPlot.ax1.text(x_mid,y_mid,txt_string,va='center',ha='center',rotation=30)

def plotTree(my_tree,parent_pt,node_txt):
    '''
    绘制决策树
    :param my_tree: 决策树（字典）
    :param parent_pt: 标注的位置
    :param node_txt: 节点名
    :return:
    '''
    decision_node=dict(boxstyle='sawtooth',fc='0.8')    #设置节点格式
    leaf_node=dict(boxstyle='round4',fc='0.8')          #设置叶子节点格式
    num_leafs=getNumLeafs(my_tree)                      #叶子节点个数，决定了树的宽度
    depth=getTreeDepth(my_tree)                         #层数
    first_str=next(iter(my_tree))
    cntr_pt=(plotTree.xOff + (1.0+float(num_leafs))/2.0/plotTree.totalW,plotTree.yOff)
    #中心位置
    plotMidtext(cntr_pt,parent_pt,node_txt)     #标注有向边的属性值
    plotNode(first_str,cntr_pt,parent_pt,decision_node) #绘制节点
    second_dict=my_tree[first_str]                      #找下一个节点
    plotTree.yOff=plotTree.yOff - 1.0/plotTree.totalD       #y偏移 （向下绘制）
    for key in second_dict.keys():
        if type(second_dict[key]).__name__=='dict':
            #若为字典，则递归绘制下一个分支树
            plotTree(second_dict[key],cntr_pt,str(key))
        else:
            #若不是字典，则绘制叶节点
            plotTree.xOff=plotTree.xOff+ 1.0/plotTree.totalW    #x偏移 （向旁绘制）
            plotNode(second_dict[key],(plotTree.xOff,plotTree.yOff),cntr_pt,leaf_node)
            #节点
            plotMidtext((plotTree.xOff,plotTree.yOff),cntr_pt,str(key))
            #边
    plotTree.yOff=plotTree.yOff+1.0/plotTree.totalD #y偏移

def createPlot(my_tree):
    #创建绘制面板
    fig=plt.figure(1,facecolor='white') #创建图像，设置格式
    fig.clf()               #清空图像
    axprops=dict(xticks=[],yticks=[])
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops) #去掉x，y轴
    plotTree.totalW=float(getNumLeafs(my_tree))     #树的宽度
    plotTree.totalD=float(getTreeDepth(my_tree))    #树的深度
    plotTree.xOff=-0.5/plotTree.totalW      #最初的x位置
    plotTree.yOff=1.0                       #最初的y位置
    plotTree(my_tree,(0.5,1.0),'')          #开始绘制决策树
    plt.show()  #显示图

def classify(input_tree,feat_labels,test_vec):
    #测试算法，执行分类
    first_str=next(iter(input_tree))    #取得第一个特征项名（到达第一个节点）
    second_dict=input_tree[first_str]   #取得该节点的划分结果
    feat_index=feat_labels.index(first_str)  #取得特征项下标
    for key in second_dict.keys():
        if test_vec[feat_index]==key:   #找到测试项所在的分支
            if type(second_dict[key]).__name__=='dict': #判断结果是否为字典（是否还有分支）
                class_label=classify(second_dict[key],feat_labels,test_vec) #是字典，到达下一层分支，递归分类
            else:
                class_label=second_dict[key]    #不是字典，得到分类结果
    return class_label

def storeTree(input_tree,filename):
    fw=open(filename,'wb')
    pickle.dump(input_tree,fw)
    fw.close()

def grabTree(filename):
    fr=open(filename,'rb')
    return pickle.load(fr)



def listOfTrees():
    tree_list = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
               {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yese'}}, 1: 'yes'}}}}]
    return tree_list


if __name__=='__main__':
    with open('lenses.txt', 'r') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lenses_labels=['age','prescripe','astigmatic','tearRate']
    lenses_tree=createTree(lenses,lenses_labels)
    print(lenses_tree)
    createPlot(lenses_tree)




