import re
from svm import *

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


def testDigits(kTup=('rbf', 10),test_rate=0.15):
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
    b, alphas = smoP(train_set,train_labels, 200, 0.0001, 10000, kTup)
    trainMat = mat(train_set)
    labelMat = mat(train_labels).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = trainMat[svInd]
    labelSV = labelMat[svInd]
    print('The test rate is %.2f and the ktup set number is %f' % (test_rate,kTup[1]))
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(trainMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, trainMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(label_list[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))
    errorCount = 0
    test_mat = mat(test_set)
    labelMat = mat(test_labels).transpose()
    m, n = shape(test_mat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, test_mat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(test_labels[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))

def multi_test():
    test_rates=[0.15,0.3,0.5]
    sets=[0.1,5,10,50,100]
    count=0
    for i in test_rates:
        for j in sets:
            count+=1
            print('Runing:%d ...' % count)
            testDigits(kTup=('rbf', j),test_rate=i)



