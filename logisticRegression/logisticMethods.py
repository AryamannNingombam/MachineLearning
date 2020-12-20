import numpy as np
import pandas as pn
from math import floor


class LogisticRegressionMethods():
    
    
    def addOne(self,z):
        dim = z.shape
        temp = np.ones([dim[0],dim[1] + 1])
        temp[...,1:] = z
        return temp
    
    
    def convertColsToRows(self,arr):
        dim = arr.shape 
        return arr.reshape([dim[0],1])    
    
    def removeBlanks(self,z):
        temp = pn.DataFrame(z)
        return np.array(temp.dropna()) 

    def splitDataset(self,z):
        splittingLength = floor(z.shape[0] * .7)
        return [z[:splittingLength],z[splittingLength:]]

    def returnXY(self,z):
        dim = z.shape
        Y = z[...,-1]
        X = z[...,:dim[1]-1]
        return [X,self.convertColsToRows(Y)]

    def sigmoid(self,z):
        dim = z.shape
        result = np.ones(dim)
        for i in range(dim[0]):
            for j in range(dim[1]):
                result[i][j] = (1/(1 + np.exp(-z[i][j])))
        return result

    def returnBinaryResult(self,z):
        dim = z.shape
        for i in range(dim[0]):
            for j in range(dim[1]):
                if z[i][j]>=0.5:
                    z[i][j] = 1
                else:
                    z[i][j] = 0
        return z
    
    def gradientDescent(self,X,Y,theta,alpha,iterations):
        m = Y.shape[0]
        for i in range(iterations):
            theta = theta - (alpha/m) * ( np.matmul( X.T, self.sigmoid(np.matmul(X,theta)) - Y) ) 
        return theta

    def scaleDown(self,X):
        means = []
        stds = []
        dim = X.shape
        result = np.ones(dim)
        for i in range(dim[1]):
            tempMean = np.mean(X[...,i])
            tempStd = np.std(X[...,i])
            tempResult =self.convertColsToRows((X[...,i] - tempMean)/tempStd)

            result[...,i] = tempResult[...,0]

            means.append(tempMean)
            stds.append(tempStd)
                
        return [result,np.array(means),np.array(stds)]