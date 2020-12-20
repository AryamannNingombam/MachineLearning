import numpy as np
import pandas as pn
from math import e,pow,log,sin
from matplotlib import pyplot as plot

def sigmoid(z):
    dim = z.shape
    result = np.ones(dim)
    for i in range(dim[0]):
        for j in range(dim[1]):
            result[i][j] = (1/(1 + (pow(e,-z[i][j]))))

    return result

def addOne(z):
    dim = z.shape
    result = np.ones([dim[0],dim[1] + 1])
    result[...,1:] = z
    return result



def cost(X,Y,theta):
    sigmoidResult = sigmoid(np.matmul(X,theta))
    m = Y.shape[0]
    result = -(1/m)*( np.matmul(Y.T,log(sigmoidResult)) + np.matmul(1-Y.T,log(1-sigmoidResult))     )
    return result

def removeLastColumn(arr):
    dim = arr.shape
    return arr[...,:dim[1]-1]

def convertColsToRows(arr):
    dim = arr.shape 
    return arr.reshape([dim[0],1])

def gradientDescent(X,Y,theta,alpha,iterations):
    m = Y.shape[0]
    for i in range(iterations):
        sigmoidResult = sigmoid(np.matmul(X,theta))
        theta = theta - (alpha/m)*(np.matmul(X.T,(sigmoidResult-Y)))

    return theta

def computeBinaryResult(result):
    dim = result.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            if (result[i][j]>=0.5):
                result[i][j] = 1
            else:
                result[i][j] = 0

    return result

def scaleDown(z):
    means = []
    stds = []
    dim = z.shape
    for i in range(dim[1]):
        means.append(np.mean(z[...,i]))
        stds.append(np.std(z[...,i]))
        z[...,i] = convertColsToRows(((z[...,i])-means[i])/stds[i])[...,0]
    return [z,(np.array(means)),(np.array(stds))]




fullDataset = pn.read_csv("./diabetesDataset/dataset.csv").values


Y = convertColsToRows(np.array(fullDataset[...,-1]))
m = Y.shape[0]

[X,means,stds] = scaleDown(removeLastColumn(fullDataset))
theta = np.ones([X.shape[1] + 1,1])

X = addOne(X)

theta = gradientDescent(X,Y,theta,0.003,20000)
totalTestCases = 500
print("Testing for 300 testcases")
allTestCaseParameters = (X[:totalTestCases,...])
allTestCaseResults = Y[:totalTestCases,...]

testCasesResults = computeBinaryResult(sigmoid(np.matmul(allTestCaseParameters,theta)))

rightTestCases = 0
for i in range(totalTestCases):
    actual = allTestCaseResults[i]
    calculated = testCasesResults[i]
    if (actual == calculated):
        rightTestCases +=1
    print(f"Test {i+1} : Expected -> {actual} Result->{calculated}")

print(f"Total test cases : {totalTestCases}, Correct : {rightTestCases} , Wrong : {totalTestCases-rightTestCases}"  )


print(f"Accuracy : {(rightTestCases/totalTestCases) * 100}%")
