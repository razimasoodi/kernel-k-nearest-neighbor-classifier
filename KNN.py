#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
from time import time


# In[2]:


#'BreastTissue'
data1= np.loadtxt(r'BreastTissue.txt',delimiter='\t',dtype=str)
data1= data1.astype(float)
for i in range(data1.shape[1]-1):
    data1[ : ,i]=(data1[ : ,i]-data1[ : ,i].mean())/data1[ : ,i].std()
x_train1,x_test1,y_train1,y_test1=train_test_split(data1[ : , :-1],data1[ : ,-1],test_size=0.3)


# In[3]:


#Diabetes
data= np.loadtxt(r'Diabetes.txt',delimiter='\t',dtype=str)
data2=data[ : , :-1]
data2= data2.astype(float)
for i in range(data2.shape[1]-1):
    data2[ : ,i]=(data2[ : ,i]-data2[ : ,i].mean())/data2[ : ,i].std()
x_train2,x_test2,y_train2,y_test2=train_test_split(data2[ : , :-1],data2[ : ,-1],test_size=0.3)


# In[4]:


#'Glass'
dataa= np.loadtxt(r'Glass.txt',delimiter='\t',dtype=str)
data3=dataa[ : , :-1]
data3= data3.astype(float)
for i in range(data3.shape[1]-1):
    data3[ : ,i]=(data3[ : ,i]-data3[ : ,i].mean())/data3[ : ,i].std()
x_train3,x_test3,y_train3,y_test3=train_test_split(data3[ : , :-1],data3[ : ,-1],test_size=0.3)


# In[5]:


#Ionosphere
daata= np.loadtxt(r'Ionosphere.txt',delimiter='\t',dtype=str)
a=[]
for i in daata:
    a.append(i.split(','))
data4=np.array(a)   
for i in range(len(data4)):
    if data4[i,-1]=='g':
        data4[i,-1] = 1
    else:
        data4[i,-1] = -1
data4= data4.astype(float)
for i in range(data4.shape[1]-1):
    #z=(data4[ : ,i]-data4[ : ,i].mean())/data4[ : ,i].std()
    data4[ : ,i]=(data4[ : ,i]-data4[ : ,i].mean())/data4[ : ,i].std()
data4=np.where(np.isfinite(data4), data4, 0)
x_train4,x_test4,y_train4,y_test4=train_test_split(data4[ : , :-1],data4[ : ,-1],test_size=0.3)


# In[6]:


#Sonar
daataa= np.loadtxt(r'Sonar.txt',delimiter='\t',dtype=str)
aa=[]
for i in daataa:
    aa.append(i.split(','))
data5=np.array(aa)   
for i in range(len(data5)):
    if data5[i,-1]=='R':
        data5[i,-1] = 1
    else:
        data5[i,-1] = -1
data5= data5.astype(float)
for i in range(data5.shape[1]-1):
    data5[ : ,i]=(data5[ : ,i]-data5[ : ,i].mean())/data5[ : ,i].std()
x_train5,x_test5,y_train5,y_test5=train_test_split(data5[ : , :-1],data5[ : ,-1],test_size=0.3)


# In[7]:


#Wine
dat= np.loadtxt(r'Wine.txt',delimiter='\t',dtype=str)
aaa=[]
for i in dat:
    aaa.append(i.split(','))
data6=np.array(aaa)
data6= data6.astype(float)
for i in range(data6.shape[1]-1):
    data6[ : ,i]=(data6[ : ,i]-data6[ : ,i].mean())/data6[ : ,i].std()
x_train6,x_test6,y_train6,y_test6=train_test_split(data6[ : , :-1],data6[ : ,-1],test_size=0.3)


# In[8]:


def one_NN(xtrain,xtest,ytrain,ytest):
    start = time()
    pre=np.zeros(ytest.shape)
    for i in range(xtest.shape[0]):
        dist=np.linalg.norm(xtrain-xtest[i].reshape((1,-1)),axis=1)  ##b tavane 2 nazashtm
        andis=np.argmin(dist)
        pre[i]=ytrain[andis]
    acc=accuracy_score(ytest,pre) 
    end=time()
    t=end-start
    return acc*100,t


# In[9]:


def linear_1NN(xtrain,xtest,ytrain,ytest):
    start = time()
    pre=np.zeros(ytest.shape)
    for i in range(xtest.shape[0]):
       # x=1+xtrain.T@xtrain
        x= 1 + (np.linalg.norm(xtrain,axis=1)**2)
        y= 1 + (np.linalg.norm(xtest[i])**2)
        #y=1+xtest[i].reshape((1,-1)).T@xtest[i].reshape((1,-1))
        #print('x',x.shape)
        #print('y',y.shape)
        z= 1 + xtrain @ xtest[i]
        dist= x + y -2*z
        andis=np.argmin(dist)
        pre[i]=ytrain[andis]
    acc=accuracy_score(ytest,pre) 
    end=time()
    t=end-start
    return acc*100,t


# In[10]:


def polynomial_1NN(xtrain,xtest,ytrain,ytest,d):    # alpha dre?????
    start=time()
    pre=np.zeros(ytest.shape)
    for i in range(xtest.shape[0]):
        x= (1 + (np.linalg.norm(xtrain,axis=1))**2)**d
        y= (1 + (np.linalg.norm(xtest[i]))**2)**d
        z= (1 + xtrain @ xtest[i])**d
        dist= x + y -2*z
        andis=np.argmin(dist)
        pre[i]=ytrain[andis]
    acc=accuracy_score(ytest,pre) 
    end=time()
    t=end-start
    return acc*100,t
        


# In[11]:


def RBF_1NN(xtrain,xtest,ytrain,ytest,sigma):
    start=time()
    pre=np.zeros(ytest.shape)
    for i in range(xtest.shape[0]):
        x=np.exp(-1*np.linalg.norm(xtrain - xtrain, axis=1) / (2 * (sigma ** 2)))
        #print('x=',x)
        y=np.exp(-1*np.linalg.norm(xtest[i].reshape((1,-1)) - xtest[i].reshape((1,-1)), axis=1)/ (2 * (sigma ** 2)))
        #print('y=',y)
        z=x=np.exp(-1*np.linalg.norm(xtrain - xtest[i].reshape((1,-1)), axis=1)**2 / (2 * (sigma ** 2)))
        dist= x+y-2*z
        andis=np.argmin(dist)
        pre[i]=ytrain[andis]
    acc=accuracy_score(ytest,pre) 
    end=time()
    t=end-start
    return acc*100 ,t    


# In[12]:


datasets=['BreastTissue','Diabetes','Glass','Ionosphere','Sonar','Wine']
Xtests=[x_test1,x_test2,x_test3,x_test4,x_test5,x_test6]
Ytests=[y_test1,y_test2,y_test3,y_test4,y_test5,y_test6]
Xtrains=[x_train1,x_train2,x_train3,x_train4,x_train5,x_train6]
Ytrains=[y_train1,y_train2,y_train3,y_train4,y_train5,y_train6]  


# In[13]:


# find best sigma
sigma_list=[1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
si=np.zeros((len(sigma_list),len(datasets)))
best_sigma=[]
for j in range(len(sigma_list)):
    for i in range(len(datasets)):
        si[j]=RBF_1NN(Xtrains[i],Xtests[i],Ytrains[i],Ytests[i],sigma_list[j])[0]
q=list(np.argmax(si,axis=0)) 
for j in range(len(q)):
    best_sigma.append(sigma_list[q[j]])


# In[14]:


NN_list=[]
linear=[]
rbf=[]
d1=[]
d2=[]
d3=[]
t1=[]
t2=[]
t3=[]
t4=[]
t5=[]
t6=[]
for i in range(len(datasets)):
    NN_list.append(one_NN(Xtrains[i],Xtests[i],Ytrains[i],Ytests[i])[0])
    t1.append(one_NN(Xtrains[i],Xtests[i],Ytrains[i],Ytests[i])[1])
    linear.append(linear_1NN(Xtrains[i],Xtests[i],Ytrains[i],Ytests[i])[0])
    t2.append(linear_1NN(Xtrains[i],Xtests[i],Ytrains[i],Ytests[i])[1])
    rbf.append(RBF_1NN(Xtrains[i],Xtests[i],Ytrains[i],Ytests[i],best_sigma[i])[0])
    t3.append(RBF_1NN(Xtrains[i],Xtests[i],Ytrains[i],Ytests[i],best_sigma[i])[1])
    d1.append(polynomial_1NN(Xtrains[i],Xtests[i],Ytrains[i],Ytests[i],1)[0])
    t4.append(polynomial_1NN(Xtrains[i],Xtests[i],Ytrains[i],Ytests[i],1)[1])
    d2.append(polynomial_1NN(Xtrains[i],Xtests[i],Ytrains[i],Ytests[i],2)[0])
    t5.append(polynomial_1NN(Xtrains[i],Xtests[i],Ytrains[i],Ytests[i],2)[1])
    d3.append(polynomial_1NN(Xtrains[i],Xtests[i],Ytrains[i],Ytests[i],3)[0])
    t6.append(polynomial_1NN(Xtrains[i],Xtests[i],Ytrains[i],Ytests[i],3)[1])
    


# In[15]:


df_marks = pd.DataFrame({'Dataset':datasets,'1NN':NN_list,'1NN+Linearkernel': linear,'1NN+RBFkernel': rbf,'1NN+Polynomialkernel (𝑑 = 1)':d1,'1NN+Polynomialkernel (𝑑 = 2)':d2,'1NN+Polynomialkernel (𝑑 = 3)':d3})
df_marks_time = pd.DataFrame({'Dataset':datasets,'1NN':t1,'1NN+Linearkernel': t2,'1NN+RBFkernel': t3,'1NN+Polynomialkernel (𝑑 = 1)':t4,'1NN+Polynomialkernel (𝑑 = 2)':t5,'1NN+Polynomialkernel (𝑑 = 3)':t6})


# In[16]:


df_marks


# In[17]:


df_marks_time

