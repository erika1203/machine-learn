#!usr/bin/env python3
# coding:utf-8
from numpy import *

def load_dataset():
    #创建一个实验样本
    posting_list=[['my','dog','has','flea','problems','help','please'],
                  ['maybe','not','take','him','to','dog','park','stupid'],
                  ['my','dalmation','is','so','cute','I','love','him'],
                  ['stop','posting','stupid','worthless','garbage'],
                  ['mr','licks','ate','my','steak','how','to','stop','him'],
                  ['quit','buying','worthless','dog','food','stupid']]
    #列表为进行词条切分后的文档集合，已去标点
    class_vec=[0,1,0,1,0,1]   #对上述样本列表的分类，按是否含侮辱性词汇，人工标注以训练程序
    return posting_list,class_vec

def create_vocab_list(data_set):
    #创建一个包含所有文档中出现的不重复词列表
    vocab_set=set([])   #创建空集合
    for document in data_set:
        #添加新词
        vocab_set=vocab_set | set(document) #|符号用作求两个集合的并集，按位或操作
    return list(vocab_set)

def set_of_words2vec(vocab_list, input_set):
    #对文档中用词是否出现在指定词汇表中进行检测
    return_vec=[0]*len(vocab_list)  #创建一个与词汇表等长的全0列表
    for word in input_set:
        if word in vocab_list:  #若词汇出现，则标识为1
            return_vec[vocab_list.index(word)]=1
        else:
            print('The word: %s is not in my vocabulary' % word)
    return return_vec

def bag_of_words2vec(vocab_list, input_set):
    # 对文档中用词出现在指定词汇表中的次数进行检测
    # 将词集模型set_of words改为词袋模型bag_of_words
    return_vec=[0]*len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)]+=1
    return return_vec

def train_nb0(train_mat,train_class):
    '''
    训练算法
    :param train_mat: 文档矩阵，是0-1矩阵
    :param train_class: 类别向量，是0-1列表
    :return: 返回两个向量，一个概率
    '''
    num_train_docs=len(train_mat)   #文档元素（行）
    num_words=len(train_mat[0]) #元素个数（列）
    p_abus=sum(train_class)/float(num_train_docs)   #侮辱性词汇占总词汇的概率
    # p_0num=zeros(num_words)
    # p_1num=zeros(num_words)
    p_0num=ones(num_words)  #数组向量
    p_1num=ones(num_words)
    # p_0denom=0.0
    # p_1denom=0.0
    p_0denom=2.0    #数字
    p_1denom=2.0
    #避免概率乘积为0的影响，将数组向量由0数组变为1数组，相应的denom也要变为2
    for i in range(num_train_docs): #对每一条文本进行检测
        if train_class[i]==1:   #文本中存在侮辱性词汇
            p_1num+=train_mat[i]    #该词对应位置+1
            p_1denom+=sum(train_mat[i]) #含侮辱性词汇的文本条的总词数
        else:
            p_0num+=train_mat[i]
            p_0denom+=sum(train_mat[i])
        # print(p_1num, p_1denom,p_0num, p_0denom)
    # p_1vec=p_1num/p_1denom
    # p_0vec=p_0num/p_0denom
    p_1vec = log(p_1num / p_1denom)  # 该词除以侮辱性文本总词数，得到条件概率
    p_0vec = log(p_0num / p_0denom)
    #为了避免原商太小导致下溢出成0，使用对数
    # print(p_1num,p_1denom,p_1vec,p_0num,p_0denom,p_0vec)
    return p_0vec,p_1vec,p_abus

def train_exam():
    post,classes=load_dataset()
    vocab=create_vocab_list(post)
    mat=[]
    print(post,classes,vocab,mat)
    for p in post:
        mat.append(bag_of_words2vec(vocab, p))
    return train_nb0(mat,classes)

def classify(input_vec,p_0vec,p_1vec,p_1class):
    '''
    分类算法
    :param input_vec: 要分类的向量
    :param p_0vec: 无侮辱词概率向量
    :param p_1vec: 含侮辱词概率向量
    :param p_1class: 侮辱词占总词概率值
    :return: 分类结果0或1
    '''
    p1=sum(input_vec*p_1vec) + log(p_1class)
    p0=sum(input_vec*p_0vec) + log(1-p_1class)
    if p1 > p0:
        return 1
    else:
        return 0

def testing_nb():
    '''
    便利函数，封装所有操作
    :return:
    '''
    post,classes=load_dataset()     #获得样本集
    vocab=create_vocab_list(post)   #去重
    train_mat=[]
    for p in post:
        train_mat.append(bag_of_words2vec(vocab, p))     #处理样本集，便于训练与计算
    p_0v,p_1v,p_ab=train_nb0(array(train_mat),array(classes))   #获得条件概率
    test_1=['love','my','dalmation']    #待分类的向量1
    this_doc1=array(bag_of_words2vec(vocab, test_1))     #将其转化形成以方便计算
    print(test_1,'classified as:',classify(this_doc1,p_0v,p_1v,p_ab))   #分类
    test_2=['stupid','garbage']
    this_doc2=array(bag_of_words2vec(vocab, test_2))
    print(test_2, 'classified as:', classify(this_doc2, p_0v, p_1v, p_ab))











