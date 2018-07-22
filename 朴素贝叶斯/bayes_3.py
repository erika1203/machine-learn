#!/usr/bin/env python3
# coding:utf-8
import bayes
import bayes_2
import operator
import random
from numpy import *
import feedparser

def cal_most_freq(vocab_list,full_test):
    '''
    找出出现频度最高的30个词
    :param vocab_list: 去重词集
    :param full_test: 所有词集
    :return: 最高频30个
    '''
    freq_dict={}
    for token in vocab_list:
        freq_dict[token]=full_test.count(token)     #计数
    sort_freq=sorted(freq_dict.items(),key=operator.itemgetter(1),reverse=True)    #排序
    return sort_freq[:30]

def local_words(feed1,feed0):
    #从个人广告中获取区域倾向
    doc_list=[]
    class_list=[]
    full_test=[]
    min_len=min(len(feed1['entries']),len(feed0['entries']))
    print(min_len)
    for i in range(min_len):    #每次访问一条rss源
        word_list=bayes_2.text_parse(feed1['entries'][i]['summary'])    #解析feed1得到的长字符串，返回字符串列表
        doc_list.append(word_list)      #将这次获得字符串列表放到总列表中
        full_test.extend(word_list)     #包含所有单词（可重复）
        class_list.append(1)        #标记为1（来源feed1）
        word_list=bayes_2.text_parse(feed0['entries'][i]['summary'])    #解析feed0得到的长字符串，返回字符串列表
        doc_list.append(word_list)
        full_test.extend(word_list)
        class_list.append(0)
    vocab_list=bayes.create_vocab_list(doc_list)    #得到一个去重总词集
    top30_words=cal_most_freq(vocab_list,full_test) #获得频数最高的30个词
    for pair in top30_words:        #从去重词集中去掉这30个词
        if pair[0] in vocab_list:
            vocab_list.remove(pair[0])
    training_set=list(range(2*min_len)) #训练集下标
    test_set=[] #测试集
    for i in range(20): #随机挑选20个样本作为测试集
        rand_index=int(random.uniform(0,len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])
    train_mat=[]
    train_class=[]
    for doc_index in doc_list:  #训练模型
        train_mat.append(bayes.bag_of_words2vec(vocab_list,doc_list[doc_index]))
        train_class.append(class_list[doc_index])
    p_0v,p_1v,p_spam=bayes.train_nb0(array(train_mat),array(train_class))
    error_count=0
    for doc_index in test_set:  #测试模型，计算错误
        word_vec=bayes.bag_of_words2vec(vocab_list,doc_list[doc_index])
        result=bayes.classify(word_vec,p_0v,p_1v,p_spam)
        if result != class_list[doc_index]:
            error_count+=1
    print('The error rate is:',float(error_count/len(test_set)))
    return vocab_list,p_0v,p_1v

def get_top_words(ny,sf):
    #获得最具表征性的词汇
    vocab_list,p_0v,p_1v=local_words(ny,sf) #去重且去高频词集，0、1的概率向量
    top_ny=[]
    top_sf=[]
    for i in range(len(p_0v)):  #扩充top列表
        if p_0v[i] > -6.0:
            top_sf.append((vocab_list[i],p_0v[i]))
        if p_1v[i] > -6.0:
            top_ny.append((vocab_list[i],p_1v[i]))
    sort_sf=sorted(top_sf,key=lambda pair:pair[1],reverse=True) #排序
    print('SF:')
    for item in sort_sf:    #获得最top值
        print(item[0])
    sort_ny = sorted(top_ny, key=lambda pair: pair[1], reverse=True)
    print('ny:')
    for item in sort_ny:
        print(item[0])

def apply():
    ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    local_words(ny,sf)
    get_top_words(ny,sf)


            