import re
import os
import operator
from numpy import *

def load_data():
    class_dict={'biology':-1,'computer':0,'psychology':1}
    word_list=[]
    class_list=[]
    for key in class_dict.keys():
        path='papers/{}'.format(key)
        paper_list=os.listdir(path)
        try:
            for paper in paper_list:
                f=open(path+'/'+paper,'r')
                words=re.split(r'\W*',f.read())
                words=[w for w in words if len(w)>3]
                # print(words)
                word_list.append(words)
                class_list.append(class_dict[key])
        except:
            print('Some papers missed...')
    return word_list,class_list

def create_vocab(word_list):
    vocab_set=set([])
    for vec in word_list:
        vocab_set=vocab_set | set(vec)
    return list(vocab_set)

def word2vec(vocab_list,input_words):
    ret_vec=[0]*len(vocab_list)
    for word in input_words:
        if word in vocab_list:
            ret_vec[vocab_list.index(word)]=1
        else:
            print('The word:%s not exist in vocab_list...' % word)
    return ret_vec

def train(word_mat,class_list):
    paper_num,vocab_num=shape(word_mat)
    minus_exist=ones(vocab_num)
    zero_exist=ones(vocab_num)
    plus_exist=ones(vocab_num)
    minus_prob=class_list.count(-1)/float(paper_num)
    plus_prob=class_list.count(1)/float(paper_num)
    zero_prob=1-minus_prob-plus_prob
    minus_denom=2.0;zero_denom=2.0;plus_denom=2.0
    for i in range(paper_num):
        if class_list[i]==-1:
            minus_exist+=word_mat[i]
            minus_denom+=sum(word_mat[i])
        elif class_list[i]==1:
            plus_exist+=word_mat[i]
            plus_denom+=sum(word_mat[i])
        else:
            zero_exist+=word_mat[i]
            zero_denom+=sum(word_mat[i])
    minus_vec=log(minus_exist/minus_denom)
    zero_vec=log(zero_exist/zero_denom)
    plus_vec=log(plus_exist/plus_denom)
    return minus_vec,zero_vec,plus_vec,minus_prob,zero_prob,plus_prob

def classify(word_mat, class_list, input_vec):
    minus_vec, zero_vec, plus_vec, minus_prob, zero_prob, plus_prob=train(word_mat, class_list)
    minus=sum(input_vec*minus_vec)+log(minus_prob)
    zero=sum(input_vec*zero_vec)+log(zero_prob)
    plus=sum(input_vec*plus_vec)+log(plus_prob)
    result_dict={minus:-1,zero:0,plus:1}
    sort_result=sorted(result_dict.items(),key=operator.itemgetter(0),reverse=True)
    print('The predicated result is :',sort_result[0][1])
    return sort_result[0][1]

def test(word_list,class_list,test_rate):
    paper_num=len(word_list)
    vocab_list=create_vocab(word_list)
    test_num=int(test_rate*paper_num)
    test_set=[];test_class=[]
    train_set=[];train_class=class_list
    for i in range(paper_num):
        vec=word2vec(vocab_list,word_list[i])
        train_set.append(vec)
    for i in range(test_num):
        rand_index=int(random.uniform(0,len(train_set)))
        test_set.append(train_set[rand_index])
        test_class.append(train_class[rand_index])
        del(train_set[rand_index])
        del(train_class[rand_index])
    print('test_set:',test_set)
    print('test_num:',test_num)
    error_count=0.0
    train_mat=array(test_set)
    for i in range(test_num):
        test_vec=array(test_set[i])
        result=classify(train_mat,train_class,test_vec)
        if result != test_class[i]:
            error_count+=1
            print('The actual result is :',test_class[i])
    print('error_count:',error_count)
    error_rate=error_count/test_num
    return error_rate

def commit():
    test_rates=[0.15,0.3,0.5]
    for test_rate in test_rates:
        print('test loop 50 times.')
        err_sum=0.0
        for i in range(50):
            word_list, class_list = load_data()
            err_sum+=test(word_list,class_list,test_rate)
        avg_err=err_sum/50
        print('err_rate for (test_rate:%.2f) is %.2f' % (test_rate,avg_err))





