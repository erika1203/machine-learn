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
            label_list.append(-1)
        del(data[0])
        data_list.append(data)
    return data_list,label_list

def stump_classify(data_mat, dim, ineq, value):
    ret_arr=ones((shape(data_mat)[0],1))
    if ineq=='lt':
        ret_arr[data_mat[:, dim] <= value]=-1
    else:
        ret_arr[data_mat[:, dim] > value]=-1
    return ret_arr


def create_best_stump(data_list,label_list,D,step_num=10):
    data_mat=mat(data_list);label_mat=mat(label_list).T
    m,n=shape(data_mat)
    min_err=inf
    best_stump={};best_result=[]
    for dim in range(n):
        max_val=data_mat[:,dim].max()
        min_val=data_mat[:,dim].min()
        step_val=(max_val-min_val)/float(step_num)
        for i in range(-2,int(step_num)+1):
            thresh_val=min_val+i*step_val
            for ineq in ['lt','gt']:
                err_arr=mat(ones((m,1)))
                result_arr=stump_classify(data_mat,dim,ineq,thresh_val)
                err_arr[result_arr==label_mat]=0
                weighted_err=D.T*err_arr
                if weighted_err < min_err:
                    min_err=weighted_err
                    best_result=result_arr.copy()
                    best_stump['dim']=dim
                    best_stump['thresh']=thresh_val
                    best_stump['ineq']=ineq
    # print('min error:',min_err)
    return best_stump,min_err,best_result

def train(data_list,label_list,iter_num=40):
    classifiers=[]
    m=len(data_list)
    D=mat(ones((m,1))/m)
    err_class=mat(zeros((m,1)))
    for i in range(iter_num):
        best_stump,error,best_result=create_best_stump(data_list,label_list,D)
        alpha=float(0.5*log10((1-error)/max(error,1e-16)))
        best_stump['alpha']=alpha
        classifiers.append(best_stump)
        # print('add stump:',best_stump)
        expon=multiply(-1*alpha*mat(label_list).T,best_result)
        D=multiply(D,exp(expon))
        D=D/sum(D)
        # print('D:',D)
        err_class+=alpha*best_result
        err_agg=multiply(sign(err_class)!=mat(label_list).T,ones((m,1)))
        err_rate=sum(err_agg)/m
        # print('The error rate for total classifiers is:',err_rate)
        if err_rate==0.0:break
    print("classifiers'number:",len(classifiers))
    return classifiers

def adaboost_classify(input_data,classifiers):
    data_mat=mat(input_data)
    agg_result=mat(zeros((shape(data_mat)[0],1)))
    for i in range(len(classifiers)):
        result=stump_classify(data_mat,classifiers[i]['dim'],classifiers[i]['ineq'],classifiers[i]['thresh'])
        agg_result+=result*classifiers[i]['alpha']
    return sign(agg_result)

def test(test_rate=0.15,iter_num=20):
    err_rates = 0.0
    for i in range(iter_num):
        data_list, label_list = load_data()
        data_num = len(data_list)
        test_num = int(data_num * test_rate)
        test_set = [];test_labels = []
        train_set = data_list;train_labels = label_list
        for i in range(test_num):
            rand_index = int(random.uniform(0, len(train_set)))
            test_set.append(train_set[rand_index])
            test_labels.append(train_labels[rand_index])
            del (train_set[rand_index])
            del (train_labels[rand_index])
        err_count = 0.0
        classifiers=train(train_set,train_labels)
        for i in range(test_num):
            result = adaboost_classify(test_set[i],classifiers)
            if result != test_labels[i]:
                err_count += 1
        err_rates += err_count / test_num
    avg_err = err_rates / iter_num
    print('The average error rate for (test rate: %.2f), (iter times:%d) is %.2f' % (test_rate, iter_num, avg_err))
    return avg_err

def commit():
    test_rates=[0.15,0.3,0.5]
    iter_nums = [30, 50, 100]
    for i in test_rates:
        for j in iter_nums:
            test(i, j)








