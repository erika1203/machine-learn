'''
2018-7-27
机器学习实战-C8.回归
'''



from numpy import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

'''
标准线性回归
'''
def load_data(filename):
    f=open(filename,'r')
    data_list=[];label_list=[]
    for line in f.readlines():
        line_vec=[]
        clean_line=line.strip().split('\t')
        feat_num=len(clean_line)-1
        for i in range(feat_num):
            line_vec.append(float(clean_line[i]))
        data_list.append(line_vec)
        label_list.append(float(clean_line[-1]))
    return data_list,label_list

def stand_regres(x_arr,y_arr):
    #按公式计算最优的系数
    #这里的x，y是data和labels
    x_mat=mat(x_arr)
    y_mat=mat(y_arr).T  #类别，列
    xTx=x_mat.T*x_mat   #数据，方阵
    xTy=x_mat.T*y_mat   #列
    if linalg.det(xTx)==0:  #linal.det(a)计算矩阵a的行列式
        print('This matrix is singular, cannot do inverse.')#行列式等于0，不可逆
        return
    else:
        ws=xTx.I*xTy    #列，维度为特征属性个数
        return ws

def plot_stand(x_arr,y_arr):
    x_mat=mat(x_arr);y_mat=mat(y_arr)
    ws=stand_regres(x_arr,y_arr)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(x_mat[:,1].flatten().A[0],y_mat.T[:,0].flatten().A[0],s=2,c='red')
    x_copy=x_mat.copy()
    x_copy.sort(0)
    y_hat=x_copy*ws
    ax.plot(x_copy[:,1],y_hat)
    plt.show()


def commit_stand():
    data_list,label_list=load_data('ex0.txt')
    ws=stand_regres(data_list,label_list)
    print('ws:',ws)
    plot_stand(data_list,label_list)

'''
加权线性回归
'''
def lwlr(x_arr,y_arr,test_p,k=1.0):
    #按公式算出ws，并与测试点相乘
    x_mat=mat(x_arr);y_mat=mat(y_arr).T
    m=shape(x_mat)[0]
    weights=mat(eye(m)) #eye(m)构造m*m的单位对角阵
    for i in range(m):
        diff_mat=test_p-x_mat[i,:]
        weights[i,i]=exp(diff_mat*diff_mat.T/(-2.0*k**2))   #计算权重矩阵W
    xTx=x_mat.T*(weights*x_mat) #加权的乘数1
    if linalg.det(xTx)==0:  #判断是否可逆
        print('This matrix is singular,cannot do inverse.')
        return
    else:
        xTy=x_mat.T*(weights*y_mat) #加权的乘数2
        ws=xTx*xTy
        print('ws:',ws) #为一个列
        print('result:',test_p*ws)
        return test_p*ws    #返回回归方程算出的预测值,为一个数

def lwlr_test(x_arr,y_arr,test_arr,k):
    #返回测试结果向量
    m=shape(test_arr)[0]
    y_hat=zeros(m)
    for i in range(m):
        y_hat[i]=lwlr(x_arr,y_arr,test_arr[i],k)    #获得所有数据的预测结果
    return y_hat    #为一个行

def plot_lwlr(x_arr,y_arr,y_hat,k):
    #画图分析模型（欠拟合和过拟合情况） 未成功
    x_mat=mat(x_arr);y_mat=mat(y_arr)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(x_mat[:, 1].flatten().A[0], y_mat.T[:, 0].flatten().A[0], s=2, c='red')
    str_ind = x_mat[:, 1].argsort(0)
    x_sort = x_mat[str_ind][:, 0, :]
    ax.plot(x_sort[:, 1], y_hat[str_ind])
    plt.title('k=%f' % k)
    plt.show()

def commit_lwlr(k=1.0):
    #参数k用来设置训练集个数
    x_list,y_list=load_data('ex0.txt')
    y_hat=lwlr_test(x_list,y_list,x_list,k) #预测结果
    print('y_hat for %f is %s' % (k,y_hat))
    plot_lwlr(x_list,y_list,y_hat,k)    #画出对应k值的拟合曲线，不成功
    plot_stand(x_list,y_list)   #画出标准线性图，成功

def rss_error(y_arr,y_hat):
    #计算测试错误（是一个连续值）
    return ((y_arr-y_hat)**2).sum()

def commit_aba():
    #预测鲍鱼的年龄
    for k in [0.1,1,10]:    #设置不同k值
        abx, aby = load_data('abalone.txt') #载入数据
        y_hat=lwlr_test(abx[0:99],aby[0:99],abx[100:199],k) #训练并测试获得结果
        print('y_hat:',y_hat)
        err=rss_error(array(aby[100:199]),y_hat)    #计算误差
        print('The error scores for k=%f is %f' % (k,err))

'''
岭回归
'''
def ridge_regres(x_mat,y_mat,lam=0.2):
    #按岭回归公式计算系数ws
    xTx=x_mat.T*x_mat
    denom=xTx+eye(shape(x_mat)[1])*lam
    if linalg.det(denom)==0:
        print('This matrix is singular,cannot do inverse.')
        return
    else:
        xTy=x_mat.T*y_mat
        ws=denom.I*xTy
        return ws

def ridge_test(x_arr,y_arr,test_num=30):
    #数据标准化并测试不同的lambda值
    x_mat=mat(x_arr);y_mat=mat(y_arr).T
    y_mean=mean(y_mat,0)
    y_mat=y_mat-y_mean
    x_mean=mean(x_mat,0)
    x_var=var(x_mat,0)
    x_mat=(x_mat-x_mean)/x_var  #标准化：减去均值后除去方差
    w_mat=zeros((test_num,shape(x_mat)[1]))
    for i in range(test_num):   #lambda成指数变化，得到不同的系数向量
        ws=ridge_regres(x_mat,y_mat,exp(i-10))
        w_mat[i,:]=ws.T
    return w_mat

def plot_ridge(weights):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(weights)
    plt.show()


def commit_ridge():
    abx,aby=load_data('abalone.txt')
    ridge_weights=ridge_test(abx,aby)
    print(ridge_weights)
    plot_ridge(ridge_weights)

'''
逐步向前回归
'''
def stage_wise(x_arr,y_arr,eps=0.01,iter_num=100):
    '''
    逐步向前回归找到最佳的系数
    :param x_arr: 数据列表
    :param y_arr: 类别列表
    :param eps: 每次迭代需要调整的步长
    :param iter_num: 迭代次数
    :return:iter*n的矩阵，包含iter次迭代各取得的最佳系数向量
    '''
    x_mat=mat(x_arr);y_mat=mat(y_arr).T
    y_mean=mean(y_mat,0)
    y_mat=y_mat-y_mean
    x_mean = mean(x_mat, 0)
    x_var = var(x_mat, 0)
    x_mat = (x_mat - x_mean) / x_var    #标准化数据
    m,n=shape(x_mat)
    ret_mat=zeros((iter_num,n))
    ws=zeros((n,1));ws_max=ws.copy()
    for i in range(iter_num):   #迭代指定次数
        print(ws.T)
        min_err=inf
        for j in range(n):  #对每一个特征的系数进行调整
            for sign in [-1,1]:
                ws_test=ws.copy()
                ws_test[j]+=eps*sign    #增加或减少指定步长值
                y_test=x_mat*ws_test    #得到回归预测结果
                rss_e=rss_error(y_mat.A,y_test.A)   #得到误差
                if rss_e < min_err: #找到误差最小的系数
                    min_err=rss_e
                    ws_max=ws_test
        ws=ws_max.copy()
        ret_mat[i,:]=ws.T
    print('The result for iter_num={0},eps={1} is {2}'.format(iter_num,eps,ret_mat))
    return ret_mat

def commit_stage():
    for iter_num in [100,300,500,1000]:
        for eps in [0.0001,0.001,0.01]:
            xa, ya = load_data('abalone.txt')
            stage_wise(xa,ya,eps,iter_num)







