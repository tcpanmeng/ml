#!/usr/bin/env python3
# -*- coding: utf-8 -*---

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
# %matplotlib inline

# Our dataset and targets


"""
加载数据集函数
@param
    fileName:文件名
@return
    dataMat 数据集
    labelMat 标签数据集
"""
def loadDataSet(fileName):
    # dataMat = []
    # labelMat = []
    # fr = open(fileName)
    # for line in fr.readlines():
    #     lineArr = line.strip().split('\t')
    #     dataMat.append([float(lineArr[0]),float(lineArr[1])])
    #     labelMat.append(float(lineArr[2]))
    dataMat = np.c_[(.4, -.7),
          (-1.5, -1),
          (-1.4, -.9),
          (-1.3, -1.2),
          (-1.1, -.2),
          (-1.2, -.4),
          (-.5, 1.2),
          (-1.5, 2.1),
          (1, 1),
          # --
          (1.3, .8),
          (1.2, .5),
          (.2, -2),
          (.5, -2.4),
          (.2, -2.3),
          (0, -2.7),
          (1.3, 2.1)].T
    labelMat = [0] * 8 + [1] * 8
    return dataMat,labelMat


# 只要函数值不等于输入值i，函数就会进行随机选择

def selectJrand(i,m):
    j = i
    while(j==i):
        j = int(rand.random.uniform(0,m))
    return j

# 调整大于H和小于L的alpha值
def clipAlpha(aj,H,L):
    if aj > H :
        aj = H
    if L >aj :
        aj = L
    return aj

# 简单SMO算法
# 输入数据，类别标签、参数C、容忍度、最大迭代数
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    pass
    # dataMatrix,labelMat = mat(dataMatIn),mat(classLabels).transpose()
    # b = 0
    # m,n = shape(dataMatrix)
    # alphas = mat(zeros((m,1)))
    # iter = 0
    # while (iter<maxIter):
    #     alphaPairsChanged = 0
    #     for i in range(m):
    #         fXi = float(multiply(alphas,labelMat).T * (dataMatrix*dataMatrix[i,:].T)) + b
    #         Ei = fXi - float(labelMat[j])
    #         alphaIold = alphas[i].copy()
    #         alphaJold = alphas[j].copy()
    #         if (labelMat[i]* Ei<-toler and alphas[i]<C)

    
def draw(x,y,clf):
    x0 = x[:,0]
    x1 = x[:,1]
    minX, maxX = min(x0), max(x0)
    minY,maxY = min(x1), max(x1)
    plt.title("matplotlib demo")
    plt.xlabel("x1")
    plt.ylabel("x2")
    colors = ['r','b']
    # plt.legend(loc='upper right')
    color = ['r' if i>0 else 'b' for i in y ]
    plt.clf()
    # print(y,color)
    plt.scatter(x0,x1,c=color,marker='o')
    plt.show()

x,y = loadDataSet('xxx')



##利用sklearn 进行学习
#选择模型
clf = svm.SVC(kernel='linear',gamma=2)
#进行测试
clf.fit(x,y)
# 支持向量
a , b = clf.support_vectors_[:, 0], clf.support_vectors_[:, 1]
print(a,b)



