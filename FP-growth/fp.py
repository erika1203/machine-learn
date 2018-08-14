'''
2018-8-14
机器学习实战
FP-growth算法
'''

class treeNode:
    #创建一个类来存放树节点
    def __init__(self,nameValue,numOccur,parentNode):
        self.name=nameValue
        self.count=numOccur
        self.nodeLink=None  #用于链接相似元素项
        self.parent=parentNode
        self.children={}    #存放子节点
    def inc(self,numOccur): #计算count
        self.count+=numOccur
    def disp(self,ind=1):   #将树以文本形式输出
         print(' '*ind,self.name,' ',self.count)
         for child in self.children.values():
             child.disp(ind+1)

def createTree(dataSet,minSup=1):
    #构建FP树，返回树节点和头指针
    headerTable={}  #头指针，存放元素值及其出现次数
    for trans in dataSet:
        for item in trans:  #第一次遍历，计算每个元素项的频数
            headerTable[item]=headerTable.get(item,0)+dataSet[trans]
    delList=[]
    for k in headerTable.keys():
        if headerTable[k] < minSup: #删除不满足最小支持度的元素项
            delList.append(k)
    for k in delList:
        del(headerTable[k])
    freqItemSet=set(headerTable.keys()) #设置频繁项集
    if len(freqItemSet)==0: #若没有频繁项，返回空
        return None,None
    for k in headerTable:
        headerTable[k]=[headerTable[k],None]   #头指针中每项增加一个值
    retTree=treeNode('Null Set',1,None) #根节点，只包含空集合
    for tranSet,count in dataSet.items():   #第二次遍历
        localD={}
        for item in tranSet:
            if item in freqItemSet: #只考虑频繁项
                localD[item]=headerTable[item][0] #存入频数
        if len(localD)>0:
            #从小到大排序
            orderedItems=[v[0] for v in sorted(localD.items(),key=lambda p:p[1],reverse=True)]
            updateTree(orderedItems,retTree,headerTable,count)
    return retTree,headerTable

def updateTree(items,inTree,headerTable,count):
    #使树生长
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]]=treeNode(items[0],count,inTree)
        if headerTable[items[0]][1]==None:
            headerTable[items[0]][1]=inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1],inTree.children[items[0]])
    if len(items)>1:
        updateTree(items[1::],inTree.children
                [items[0]],headerTable,count)

def updateHeader(nodeToTest,targetNode):
    #确保节点链接指向树中该元素项的每一个实例
    while nodeToTest.nodeLink != None:
        nodeToTest=nodeToTest.nodeLink
    nodeToTest.nodeLink=targetNode

def createInitSet(dataset):
    retDict={}
    for trans in dataset:
        retDict[frozenset(trans)]=1
    return retDict

def commit():
    simpleDat = [['r', 'z', 'h', 'j', 'p'],
                 ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                 ['z'],
                 ['r', 'x', 'n', 'o', 's'],
                 ['y', 'r', 'x', 'z', 'q', 't', 'p'],
                 ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    initset=createInitSet(simpleDat)
    minSupport=3
    print('--initset:--\n',initset)
    FPtree,header=createTree(initset,minSupport)
    print('--FPtree:--')
    print(FPtree.disp())
    print('--header:--\n',header)
    freqItemList=[]
    mineTree(FPtree,header,minSupport,set([]),freqItemList)
    print('--freqItemList:--\n',freqItemList)


'''
挖掘频繁模式
'''
def ascendTree(leafNode,prefixPath):
    #迭代上溯整棵树
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent,prefixPath)

def findPrefixPath(basePat,treeNode):
    #生成条件模式基
    condPats={}
    while treeNode != None:
        #访问树中所有包含给定元素项的节点
        prefixPath=[]
        ascendTree(treeNode,prefixPath)
        if len(prefixPath)>1:
            condPats[frozenset(prefixPath[1:])]=treeNode.count
        treeNode=treeNode.nodeLink
    return condPats

def mineTree(inTree,headerTable,minSup,prefix,freqItemList):
    #递归查找频繁项集
    # #对头指针表中的元素项按从小到大排序
    bigL=[v[0] for v in sorted(headerTable.items(),key=lambda p:p[0])]
    #将每一个频繁项集加入到频繁项集列表中
    for basePat in bigL:
        newFreqSet=prefix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases=findPrefixPath(basePat,headerTable[basePat][1])
        condTree,Head=createTree(condPattBases,minSup)
        if Head != None:
            print('Conditional tree for:',newFreqSet)
            condTree.disp(1)
            mineTree(condTree,Head,minSup,newFreqSet,freqItemList)

def example():
    parseDat=[line.split() for line in open('kosarak.dat').readlines()]
    initset=createInitSet(parseDat)
    minSupport=100000
    print('--initset:--\n', initset)
    FPtree, header = createTree(initset, minSupport)
    print('--FPtree:--')
    print(FPtree.disp())
    print('--header:--\n', header)
    freqItemList = []
    mineTree(FPtree, header, minSupport, set([]), freqItemList)
    print('--freqItemList:--\n', freqItemList)

