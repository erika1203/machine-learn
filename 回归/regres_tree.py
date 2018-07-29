'''
2018-7-28
机器学习实战 C8-树回归
'''

from numpy import *

'''
构建树
'''
def load_data(filename):
    #数据与目标变量放在一起
    data_list=[]
    fr=open(filename)
    for line in fr.readlines():
        cur_line=line.strip().split('\t')
        flt_line=list(map(float,cur_line))
        data_list.append(flt_line)
    return data_list

def bin_split(dataset,feature,value):
    '''
    将数据集合分成两个子集
    :param dataset: 数据集
    :param feature: 待切分的特征
    :param value: 按某特征值切分
    :return: 两个子集
    '''
    mat0=dataset[nonzero(dataset[:,feature] > value)[0],:]
    mat1 = dataset[nonzero(dataset[:, feature] <= value)[0], :]
    # print(mat0[0],mat1[0])
    return mat0,mat1


def reg_leaf(dataset):
    #生成叶节点，为目标变量的均值
    return mean(dataset[:,-1])

def reg_err(dataset):
    #平方误差估计（总方差）
    return var(dataset[:,-1])*shape(dataset)[0]

def choose_best_split(dataset,leaf_type=reg_leaf,err_type=reg_err,ops=(1,4)):
    #选择最佳划分方式，若找不到一个好的划分，则退出，在create_tree中构建叶节点
    tols=ops[0];toln=ops[1]         #允许误差下降值和切分的最少样本数
    if len(set(dataset[:,-1].T.tolist()[0]))==1:    #如果只有一个元素，构建叶节点
        return None,leaf_type(dataset)
    m,n=shape(dataset)
    diff=err_type(dataset)      #误差
    best_diff=inf;best_index=0;best_value=0
    for feat_index in range(n-1):   #对每一个特征（最后一列是类别，删去）
        feat_set=set()
        for vec in dataset[:,feat_index].tolist():
            feat_set.add(vec[0])
        for feat_val in feat_set: #对某特征的每一个值
            mat0,mat1=bin_split(dataset,feat_index,feat_val)    #划分两个子集
            if (shape(mat0)[0]<toln) or (shape(mat1)[0]<toln):continue  #分出子集个数太小，则跳过
            new_diff=err_type(mat0)+err_type(mat1)  #计算划分后得到的新误差
            if new_diff<best_diff:      #更新最佳划分方式
                best_index=feat_index;best_value=feat_val
                best_diff=new_diff
    if (diff-best_diff)<tols:       #如果新旧误差相差太小，则构建叶节点
        return None,leaf_type(dataset)
    mat0,mat1=bin_split(dataset,best_index,best_value)  #按最佳方式划分，若子集个数太小，则构建叶节点
    if (shape(mat0)[0]<toln) or (shape(mat1)[0]<toln):
        return None,leaf_type(dataset)
    return best_index,best_value

def create_tree(dataset,leaf_type=reg_leaf,err_type=reg_err,ops=(1,4)):
    '''
    构建树
    :param dataset: 数据集
    :param leaf_type: 给出建立叶节点的函数
    :param err_type: 误差计算函数
    :param ops: 包含树构建所需要的其他参数的元组，可用来控制树的形状
    :return: 树（字典），包含四个元素：特征下标，特征值，左子树，右子树
    '''
    feat,val=choose_best_split(dataset,leaf_type,err_type,ops)
    if feat==None:  #叶节点
        return val
    ret_tree={}
    ret_tree['sp_ind']=feat
    ret_tree['sp_val']=val
    lset,rset=bin_split(dataset,feat,val)   #划分左右子树
    ret_tree['left']=create_tree(lset,leaf_type,err_type,ops)   #递归分别建立左右子树
    ret_tree['right']=create_tree(rset,leaf_type,err_type,ops)
    return ret_tree

def leaves_cnt(tree):
    count=0
    next_dict=[tree]
    for d in next_dict:
        for key in d.keys():
            if key=='left' or key=='right':
                if type(d[key]) != dict:
                    count+=1
                else:
                    next_dict.append(d[key])
    return count

def depth(tree):
    if len(tree)==0:
        return 0
    d1=d2=0
    if 'left' in tree.keys():
        if type(tree['left'])==dict:
            d1=depth(tree['left'])
        else:
            d1=1
    if 'right' in tree.keys():
        if type(tree['right'])==dict:
            d2=depth(tree['right'])
        else:
            d2=1
    return 1+max(d1,d2)

def commit_create():
    dataset=load_data('ex000.txt')
    print(dataset)
    tree=create_tree(mat(dataset))
    print(tree)
    leaves=leaves_cnt(tree)
    depths=depth(tree)
    print('The leaves of this tree:',leaves)
    print('The depth of this tree:',depths)

'''
后剪枝
'''
def is_tree(obj):
    return type(obj)==dict

def get_mean(tree):
    #递归方法找到两个叶节点，计算它们的平均值
    if is_tree(tree['right']):
        tree['right']=get_mean(tree['right'])
    if is_tree(tree['left']):
        tree['left']=get_mean(tree['left'])
    # print('view tree:',tree)
    return (tree['right']+tree['left'])/2.0

def prune(tree,testset):
    #对树进行剪枝
    if shape(testset)[0]==0:    #判断是否为空
        return get_mean(tree)
    if is_tree(tree['right']) or is_tree(tree['left']): #递归，对测试数据进行切分
        lset, rset = bin_split(testset, tree['sp_ind'], tree['sp_val'])
        if is_tree(tree['left']):   #若是子树，递归对其剪枝
            tree['left']=prune(tree['left'],lset)
        if is_tree(tree['right']):
            tree['right']=prune(tree['right'],rset)
    if not is_tree(tree['right']) and not is_tree(tree['left']):   #若不是子树，则判断是否合并
        lset, rset = bin_split(testset, tree['sp_ind'], tree['sp_val']) #切分
        # print(lset,type(lset),tree['left'],type(tree['left']))
        # print(rset, type(rset), tree['right'], type(tree['right']))
        for key in ['left','right']:
            if tree[key]==None:
                tree[key]=0
        init_error=sum(power(lset[:,-1]-tree['left'],2)) \
                    + sum(power(rset[:,-1]-tree['right'],2))    #计算合并前的误差
        tree_mean=(tree['right']+tree['left'])/2.0
        merge_error=sum(power(testset[:,-1]-tree_mean,2))       #计算合并后的误差
        if merge_error < init_error:        #判断是否合并
            print('merging...')
            return tree_mean
        else:
            return tree
    return tree

def commit_prune():
    trainset=load_data('ex2.txt')
    tree=create_tree(mat(trainset),ops=(0,1))
    print('init tree:', tree)
    print('leaves:',leaves_cnt(tree))
    print('depths:',depth(tree))
    # mean=get_mean(tree)
    # return mean
    testset=load_data('ex2test.txt')
    # print('testset:',testset)
    merge_tree=prune(tree,mat(testset))
    print('merge tree:',merge_tree)
    print('leaves:', leaves_cnt(merge_tree))
    print('depths:', depth(merge_tree))

'''
模型树
'''
def linear_solve(dataset):
    #将数据集转化成自变量x和目标变量y，并计算标准回归系数
    m,n=shape(dataset)
    x=mat(ones((m,n)));y=mat(ones((m,1)))
    x[:,1:n]=dataset[:,0:n-1];y=dataset[:,-1]   #x留出第一列全设置为1，以作为常数项
    xTx=x.T*x
    if linalg.det(xTx)==0.0:
        raise(NameError('This matrix is singular,cannot do inverse,\n\
                        try increasing the second value of ops.'))
    ws=xTx.I*(x.T*y)
    return ws,x,y

def model_leaf(dataset):
    #生成模型树的叶节点
    ws,x,y=linear_solve(dataset)
    return ws       #叶节点存储标准回归系数

def model_err(dataset):
    #计算模型的平方误差
    ws,x,y=linear_solve(dataset)
    yhat=x*ws
    return sum(power(y-yhat,2))

def commit_model():
    dataset=load_data('exp2.txt')
    tree=create_tree(mat(dataset),model_leaf,model_err,(1,10))
    print(tree)
    print('leaves:', leaves_cnt(tree))
    print('depths:', depth(tree))
    return tree

'''
树回归测试与比较
'''
def regtree_eval(model,indat):
    #返回回归树的叶节点值,indat是为了与函数modeltree_eval保持一致
    return float(model)

def modeltree_eval(model,indat):
    #返回模型树叶节点中的ws与x相乘后得出的yhat
    n=shape(indat)[1]
    x=mat(ones((1,n+1)))
    x[:,1:n+1]=indat
    return float(x*model)

def tree_forecast(tree,indata,model_eval):
    #自顶向下遍历树，直到叶子节点，model_eval用来调用回归树或模型树进行预测
    if not is_tree(tree):return model_eval(tree,indata)
    if indata[tree['sp_ind']]>tree['sp_val']:
        if is_tree(tree['left']):
            return tree_forecast(tree['left'],indata,model_eval)
        else:
            return model_eval(tree['left'],indata)
    else:
        if is_tree(tree['right']):
            return tree_forecast(tree['right'], indata,model_eval)
        else:
            return model_eval(tree['right'], indata)

def create_forecast(tree,testset,model_eval):
    #对测试集进行测试，将结果存到矩阵中
    m=shape(testset)[0]
    yhat=mat(zeros((m,1)))
    for i in range(m):
        yhat[i,0]=tree_forecast(tree,mat(testset[i]),model_eval)
    return yhat

def test():
    #测试回归树、模型树和标准回归的执行结果，用R**2值进行比较
    trainset=mat(load_data('bikeSpeedVsIq_train.txt'))
    testset = mat(load_data('bikeSpeedVsIq_test.txt'))
    print('regression tree...') #开始进行回归树测试
    regtree=create_tree(trainset,ops=(1,20))    #构建回归树
    leaves=leaves_cnt(regtree)
    depths=depth(regtree)
    print(regtree,'\n',leaves,depths)
    yhat=create_forecast(regtree,testset[:,0],regtree_eval)     #得到测试结果
    cor=corrcoef(yhat,testset[:,1],rowvar=False)[0,1]           #计算R**2值
    print('corrcoef:',cor)
    print('model tree...')      #开始进行模型树测试
    modeltree = create_tree(trainset, model_leaf,model_err,ops=(1, 20)) #构建模型树
    leaves = leaves_cnt(modeltree)
    depths = depth(modeltree)
    print(modeltree, '\n', leaves, depths)
    yhat = create_forecast(modeltree, testset[:, 0], modeltree_eval)    #得到测试结果
    cor = corrcoef(yhat, testset[:, 1], rowvar=False)[0, 1] #获得R**2值
    print('corrcoef:', cor)
    print('standart regression...')     #开始测试标准回归
    ws,x,y=linear_solve(trainset)       #得到标准回归的系数，自变量，目标变量
    print(ws)
    for i in range(shape(testset)[0]):  #得到预测结果yhat
        yhat[i]=testset[i,0]*ws[1,0]+ws[0,0]
    cor = corrcoef(yhat, testset[:, 1], rowvar=False)[0, 1]  # 获得R**2值
    print('corrcoef:', cor)



















