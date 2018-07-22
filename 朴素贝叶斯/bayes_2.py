#!usr/bin/env python3
# coding:utf-8

import bayes
import re
from numpy import *

def split_text():
    #将字符串分词处理的一个例子
    my_sent='This book is the best book on Python or M.L. I have ever laid eye upon.'
    regex=re.compile('\\W*')    #定义一个匹配规则
    #此处的规则是把符合'\\W*'的地方当作分割点，即在非数字、字母字符的地方分割开，不留下标点符号
    print(my_sent.split())      #查看原始方法的结果
    list_of_tokens=regex.split(my_sent)     #对regex处进行分割
    print(list_of_tokens)       #查看效果
    list_of_toks=[tok.lower() for tok in list_of_tokens if len(tok)>0]  #去空值，使每个单词都是小写
    print(list_of_toks)         #查看效果

def text_parse(big_string):
    #解析文本成字符串列表
    list_of_tokens=re.split(r'\W*',big_string)  #分割
    return [tok.lower() for tok in list_of_tokens if len(tok)>2]    #去掉一些空值和无意义的小词，使小写

def spam_test():
    #使用贝叶斯垃圾邮件分类器
    doc_list=[]         #包含所有文件解析后的词列表（二维）
    class_list=[]
    full_text=[]        #包含所有出现词语的列表（一维）
    text_num=50
    for i in range(1,26):                #共25个文件
        try:
            big_string=open('email/spam/%d.txt' % i).read()   #导入垃圾邮件文件
            word_list=text_parse(big_string)        #解析为词列表
            doc_list.append(word_list)              #加入所有文件的总列表中
            full_text.extend(word_list)             #加入所有词语的列表中
            class_list.append(1)            #将类别设为1
        except:
            text_num-=1
        try:
            big_string = open('email/ham/%d.txt' % i).read()  #导入非垃圾邮件文件
            word_list=text_parse(big_string)
            doc_list.append(word_list)
            full_text.extend(word_list)
            class_list.append(0)
        except:
            text_num -= 1
    print(text_num)
    vocab_list=bayes.create_vocab_list(doc_list)    #去重列表（二维）
    training_set=list(range(text_num))      #全部样本集数
    test_set=[]         #测试集
    for i in range(10): #从样本集中随机挑选10个作为测试集（下标）
        rand_index=int(random.uniform(0,len(training_set))) #随机整数
        test_set.append(training_set[rand_index])   #加入测试集
        del(training_set[rand_index])       #从训练集中删掉
    train_mat=[]    #训练矩阵
    train_class=[]  #训练矩阵的元素类别
    for doc_index in training_set:
        words=bayes.set_of_words2vec(vocab_list,doc_list[doc_index])    #检测doc_list[i]中的词语是否出现在vocab_list中，返回0-1向量
        train_mat.append(words)  #放入矩阵中
        train_class.append(class_list[doc_index])   #更新类别
    p_0v,p_1v,p_spam=bayes.train_nb0(array(train_mat),array(train_class))   #求出相关概率
    error_count=0
    for doc_index in test_set:  #测试数据
        word_vec=bayes.set_of_words2vec(vocab_list,doc_list[doc_index])
        result=bayes.classify(array(word_vec),p_0v,p_1v,p_spam) #与相关概率比较得到分类结果
        if result != class_list[doc_index]:
            error_count+=1
    print('The error rate is :',float(error_count/len(test_set)))   #计算错误率
    return float(error_count/len(test_set))

def avg_error():
    #计算执行200次测试之后的平均错误率
    from functools import reduce
    error_list=[]
    for i in range(200):
        error=spam_test()
        error_list.append(error)
    def count(x,y):
        return x+y
    sum=reduce(count,error_list)
    avg_error=sum/200
    return avg_error











