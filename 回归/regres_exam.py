'''
预测乐高玩具套装的价格
'''

from numpy import *
from regression import *
from time import sleep
import requests

def load_datas():
    # response=requests.get('http://archive.ics.uci.edu/ml/machine-learning-databases/00409/Daily_Demand_Forecasting_Orders.csv')
    # text=response.text
    # fw=open('test.txt','w')
    # fw.write(text)
    data_arr=[];label_arr=[]
    fr=open('test.txt','r')
    for line in fr.readlines():
        try:
            line_arr=[1]
            cur_line=line.strip().split(';')
            label_arr.append(float(cur_line[-1]))
            for i in range(len(cur_line)-1):
                line_arr.append(float(cur_line[i]))
            data_arr.append(line_arr)
        except:
            print(line.strip())
    return data_arr,label_arr

def plot_stands(x_arr,y_arr):
    #标准回归图
    x_mat=mat(x_arr);y_mat=mat(y_arr)
    ws=stand_regres(x_arr,y_arr)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(x_mat[:,3].flatten().A[0],y_mat.T[:,0].flatten().A[0],s=2,c='red')
    x_copy=x_mat.copy()
    x_copy.sort(0)
    y_hat=x_copy*ws
    ax.plot(x_copy[:,3],y_hat)
    plt.show()

def stand_tests():
    #测试标准回归的效果
    x_arr,y_arr=load_datas()
    m=len(x_arr)
    index_list=list(range(m))
    random.shuffle(index_list)
    x_test=[];y_test=[]
    x_train=[];y_train=[]
    for i in range(m):  #随机挑选测试集和训练集
        if i < m*0.9:
            x_train.append(x_arr[index_list[i]])
            y_train.append(y_arr[index_list[i]])
        else:
            x_test.append(x_arr[index_list[i]])
            y_test.append(y_arr[index_list[i]])
    ws=stand_regres(x_train,y_train)    #得到标准回归方法下的系数
    print('The best model from Ridge Regression is \n',ws)
    plot_stands(x_train,y_train)    #画图以查看效果

def ridge_tests(iter_num=10,test_num=10):
    '''
    使用岭回归找到最佳系数
    :param iter_num: 随机检验的迭代次数
    :param test_num: 在岭回归测试中选用多少个不同的lambda值
    :return:
    '''
    x_arr, y_arr = load_datas()
    err_mat=zeros((iter_num,test_num))  #矩阵，行对应迭代次数，列对应不同的lambda值
    m = len(x_arr)
    index_list = list(range(m))     #索引列表，用作随机抽取测试集的下标
    w_mat=None
    for j in range(iter_num):       #迭代，会得到不同的测试集和训练集
        random.shuffle(index_list)  #洗乱索引列表中的值
        x_test = [];y_test = []
        x_train = [];y_train = []
        for i in range(m):      #遍历原数据集，按规则放入训练集或测试集
            if i < m * 0.9:     #90%数据作为训练集
                x_train.append(x_arr[index_list[i]])
                y_train.append(y_arr[index_list[i]])
            else:               #10%数据作为测试集
                x_test.append(x_arr[index_list[i]])
                y_test.append(y_arr[index_list[i]])
        w_mat = ridge_test(x_train, y_train, test_num)
        print('x_train:\n',x_train)
        print('x_test:\n',x_test)
        print('w_mat:\n',w_mat)
        #调用测试函数，返回不同lambda得到的ws向量组成的矩阵
        for k in range(test_num):   #对不同lambda值
            test_xmat=mat(x_test);train_xmat=mat(x_train)
            #对测试集按训练集获得的均值和方差进行标准化
            mean_train=mean(train_xmat,0)
            var_train=var(train_xmat,0)
            test_xmat=(test_xmat-mean_train)/var_train
            #得到y的估计值
            y_est=test_xmat*mat(w_mat[k,:]).T+mean(y_train)
            err_mat[j,k]=rss_error(y_est.T.A,array(y_test))
            #调用差值计算函数，计算估计值和实际值的方差
    print('err_mat:\n',err_mat)
    mean_err=mean(err_mat,0)    #获得不同lambda值各自迭代多次的平均差值
    min_err=float(min(mean_err))    #选取最小的作为最佳系数
    best_weights=w_mat[nonzero(mean_err==min_err)]
    print('best_weights with regulation:',best_weights)
    x_mat=mat(x_arr);y_mat=mat(y_arr).T
    mean_x=mean(x_mat,0);var_x=var(x_mat,0)
    unregular_weights=best_weights/var_x        #将标准化过的数据还原
    print('The best model from Ridge Regression is \n',unregular_weights)
    print('with constant term:',-1*sum(multiply(mean_x,unregular_weights))+mean(y_mat))








