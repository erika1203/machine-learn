'''
2018-7-29
使用tkinter库创建GUI（用户图形界面）
'''

from numpy import *
from tkinter import *
import regres_tree
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg     #链接matplotlib和tkagg
from matplotlib.figure import Figure

def reDraw(tols,toln):
    #绘制图
    reDraw.f.clf()  #清空图像
    reDraw.a=reDraw.f.add_subplot(111)  #创建子图
    # print(reDraw.rawDat)
    print(reDraw.testDat)
    if chkBtnVar.get(): #检查复选框是否被选中，判断画出回归树还是模型树
        if toln<2:toln=2
        mytree=regres_tree.create_tree(reDraw.rawDat,regres_tree.model_leaf,regres_tree.model_err,(tols,toln))
        print(mytree)
        yhat=regres_tree.create_forecast(mytree,reDraw.testDat,regres_tree.modeltree_eval)
        print(yhat)
    else:
        mytree = regres_tree.create_tree(reDraw.rawDat, ops=(tols,toln))
        print(mytree)
        yhat = regres_tree.create_forecast(mytree, reDraw.testDat, regres_tree.regtree_eval)
        print(yhat)
    reDraw.a.scatter(reDraw.rawDat[:,0].tolist(),reDraw.rawDat[:,1].tolist(),s=5)     #真实y值用散点图
    reDraw.a.plot(reDraw.testDat,yhat,linewidth=2.0)        #预测值用线来拟合
    reDraw.canvas.show()

def getInputs():
    #试图理解用户输入并防止程序崩溃
    try:toln=int(tolNentry.get())   #期望输入是整数
    except:     #输入错误时，用默认值代替
        toln=10
        print('enter integer for toln')
        tolNentry.delete(0,END)
        tolNentry.insert(0,'10')
    try:
        tols = int(tolNentry.get())
    except:
        tols = 1.0
        print('enter integer for tols')
        tolSentry.delete(0, END)
        tolSentry.insert(0, '1.0')
    return toln,tols

def draw_newtree():
    #生成图
    toln,tols=getInputs()
    reDraw(tols,toln)

#构建一组tkinter模块
root=Tk()   #根部件，会显示一个空白框

reDraw.f=Figure(figsize=(5,4),dpi=100)  #创建画布
reDraw.canvas=FigureCanvasTkAgg(reDraw.f,master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0,columnspan=3)

Label(root,text='Plot Place Holder').grid(row=0,columnspan=3)
#，grid方法设置行和列，columnspan表示跨列
Label(root,text='tolN').grid(row=1,column=0)
tolNentry=Entry(root)   #entry为文本输入框，允许单行文本输入
tolNentry.grid(row=1,column=1)
tolNentry.insert(0,'10')
Label(root,text='tolS').grid(row=2,column=0)
tolSentry=Entry(root)
tolSentry.grid(row=2,column=1)
tolSentry.insert(0,'1.0')
Button(root,text='ReDraw',command=draw_newtree).grid(row=1,column=2,rowspan=3)
chkBtnVar=IntVar()  #按钮整数值，用于读取checkbutton的状态
chkBtn=Checkbutton(root,text='Model tree',variable=chkBtnVar)   #复选按钮
chkBtn.grid(row=3,column=0,columnspan=2)

reDraw.rawDat=mat(regres_tree.load_data('sine.txt'))
reDraw.testDat=arange(min(reDraw.rawDat[:,0]),max(reDraw.rawDat[:,0]),0.01)
reDraw(1.0,10)

root.mainloop()

