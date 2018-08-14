'''
2018-8-13
机器学习实战 聚类算法 k-means
'''

from numpy import *

def loadDataset(filename):
    #加载数据
    datamat=[]
    fr=open(filename)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        fltLine=list(map(float,curLine))
        datamat.append(fltLine)
    return datamat

def distEclud(vecA,vecB):
    #计算欧氏距离
    return sqrt(sum(power(vecA-vecB,2)))

def randCent(dataset,k):
    #初始化随机挑选k个质心
    n=shape(dataset)[1]
    centroids=mat(zeros((k,n)))
    for j in range(n):
        minJ=min(dataset[:,j])
        maxJ=max(dataset[:,j])
        rangeJ=float(maxJ-minJ)
        centroids[:,j]=minJ+rangeJ*random.rand(k,1)
        #保证质心在各特征值范围之间，random.rand(m,n)生成m行n列的0-1内数的矩阵
    return centroids

def kmeans(dataset,k,distMea=distEclud,createCent=randCent):
    #执行kmeans算法
    m=shape(dataset)[0]
    clusterAssment=mat(zeros((m,2)))
    centroids=createCent(dataset,k)
    clusterChanged=True
    while clusterChanged: #当未收敛时
        clusterChanged=False
        for i in range(m):  #对每一个数据
            minDist=inf;minIndex=-1
            for j in range(k):  #找到该数据对应的类别
                distJI=distMea(centroids[j,:],dataset[i,:])
                if distJI<minDist:
                    minDist=distJI;minIndex=j
            if clusterAssment[i,0]!=minIndex:
                clusterChanged=True
            clusterAssment[i,:]=minIndex,minDist**2
            #Assment分配结果存储类别，距离.距离用作衡量误差
        print(centroids)
        for cent in range(k):   #重新计算质心值
            ptsInclust=dataset[nonzero(clusterAssment[:,0].A==cent)[0]]
            #找到某一类别中的所有数据，计算其均值（新的质心）
            centroids[cent,:]=mean(ptsInclust,axis=0)
    return centroids,clusterAssment

def commit1(k):
    datamat=mat(loadDataset('testSet.txt'))
    centroids,results=kmeans(datamat,k)
    print('clusering %d kinds...' % k)
    print('centroids:\n',centroids)
    print('results:\n',results)


'''
为了克服kmeans算法的收敛值为局部最优而非全局最优，需要优化算法。
二分Kmeans方法，将所有点看作一个簇，不断二分划分，直到达到指定k
'''
def binKmeans(dataset,k,distMea=distEclud):
    #二分kmeans法
    m=shape(dataset)[0]
    clusterAssment=mat(zeros((m,2)))
    centroid0=mean(dataset,axis=0).tolist()[0] #初始只有一个簇
    centList=[centroid0]
    for j in range(m):  #初始各数据到质心的距离
        clusterAssment[j,1]=distMea(mat(centroid0),dataset[j,:])**2
    while len(centList)<k:
        #在划分k个簇之前，循环
        lowestSSe=inf
        for i in range(len(centList)):  #对各个簇执行二分，找到最佳二分处
            ptsIncurrcluster=dataset[nonzero(clusterAssment[:,0].A==i)[0],]
            #对某个簇用普通k均值法一分为二
            centroidMat,splitClustAss=kmeans(ptsIncurrcluster,2,distMea)
            ssesplit=sum(splitClustAss[:,1])    #该簇二分后的误差
            sseNotSplit=sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            #其他未执行二分的簇的原有误差
            print('ssesplit,and notsplit:',ssesplit,sseNotSplit)
            if (ssesplit+sseNotSplit)<lowestSSe:
                #若二分后的误差小于原误差，则该簇需要划分
                bestCentosplit=i
                bestNewcents=centroidMat    #划分后的新簇的质心
                bestClustAss=splitClustAss.copy()   #划分后原簇各数据分配到的结果
                lowestSSe=sseNotSplit+ssesplit  #更新最小误差值
        bestClustAss[nonzero(bestClustAss[:,0].A==1)[0],0]=len(centList)
        #更新类别名字
        bestClustAss[nonzero(bestClustAss[:,0].A==0)[0],0]=bestCentosplit
        print('the best cent to split is:',bestCentosplit)  #划分的原簇
        print('the len of best cluster ass is:',len(bestClustAss))  #新分配结果
        centList[bestCentosplit]=bestNewcents[0,:]
        centList.append(bestNewcents[1,:])  #更新原簇质心和增加新簇质心
        clusterAssment[nonzero(clusterAssment[:,0].A==bestCentosplit)[0],:]=bestClustAss
        #更新分配结果
    return centList,clusterAssment

def commit2(k):
    dataset=loadDataset('testSet2.txt')
    centroids,results=binKmeans(mat(dataset),k)
    print('clusering %d kinds...' % k)
    print('centroids:\n', centroids)
    print('results:\n', results)




