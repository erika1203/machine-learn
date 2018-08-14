'''
一个利用k均值聚类算法的实例，从yahooapi中获得数据，无法打开网址
'''

from urllib import parse,request
from time import sleep
from Kmeans import *
from math import *
from numpy import *
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def geoGrad(address,city):
    #从yahooapis中读取数据
    apiStem='http://where.yahooapis.com/geocode?'
    params={}
    params['flags']='J'
    params['appid']='ppp68N8t'
    params['location']='%s %s' % (address,city)
    url_parms=parse.urlencode(params)
    yahooapi=apiStem+url_parms
    print(yahooapi)
    c=request.urlopen('yahooapi')
    return json.loads(c.read())

def massPlaceFind(filename):
    #将已有地址传到yahooapis中，返回数据，写入新文件中
    fw=open('places.txt','w')
    for line in open(filename).readlines():
        line=line.strip()
        lineArr=line.split('\t')
        retDict=geoGrad(lineArr[1],lineArr[2])
        if retDict['ResultSet']['Error']==0:
            lat=float(retDict['ResultSet']['Results'][0]['latitude'])
            lng=float(retDict['ResultSet']['Results'][0]['longitude'])
            print('%s\t%f\t%f' % (line,lat,lng))
            fw.write('%s\t%f\t%f' % (line,lat,lng))
        else:
            print('error fetching')
        sleep(1)
    fw.close()

def distSLC(vecA,vecB):
    #计算球面距离
    a=sin(vecA[0,1]*pi/180)*sin(vecB[0,1]*pi/180)
    b=cos(vecA[0,1]*pi/180)*cos(vecB[0,1]*pi/180)*cos(pi*(vecB[0,0]-vecA[0,0])/180)
    return acos(a+b)*6371

def clusterCLubs(numClust=5):
    #执行k均值算法并画图
    datList=[]  #导入数据集
    for line in open('places.txt').readlines():
        lineArr=line.split('\t')
        datList.append([float(lineArr[4]),float(lineArr[3])])
        datMat=mat(datList)
    #调用算法，获得聚类结果
    myCentroids,clustAssing=binKmeans(datMat,numClust,distSLC)
    fig=plt.figure()    #开始画图
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s','o','^','8','p','d','v','h','>','<']
    axprops=dict(xticks=[],yticks=[])
    ax0=fig.add_axes(rect,label='ax0',**axprops)
    imgP=plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect,label='ax1',frameon=False)
    for i in range(numClust):
        ptsIncurrcluster=datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle=scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsIncurrcluster[:,0].flatten().A[0],ptsIncurrcluster[:,1].flatten().A[0],\
                    marker=markerStyle,s=90,c='blue')
    ax1.scatter(myCentroids[0].flatten().A[0],myCentroids[1].flatten().A[0],marker='+',s=300,c='red')
    plt.show()

