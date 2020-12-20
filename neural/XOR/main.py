import numpy as np
from math import exp



class Pair():
    def __init__(self,first,second) -> None:
        self.first = first
        self.second=  second



class XOR():

    def __init__(self,arr,answers)->None:
        self.firstLayer = np.array(arr)
        self.target = np.array(answers)
        # self.secondLayerWeights= np.array([[
        #     -20,-20
        # ],[
        #     20,20
        # ]])
        self.secondLayerWeights = np.random.randint(-1,1,size=(2,2))
        # self.thirdLayerWeights = np.array([
        #     [20,20]
        # ])
        self.thirdLayerWeights = np.random.randint(-1,1,size=(1,2))
        # self.secondLayerBias = [[30],[-10]]
        self.secondLayerBias = np.array([[1],[1]])
        # self.thirdLayerBias = [[-30]]
        self.thirdLayerBias = np.array([[1]])

    def returnBinary(self,arr): 
        return arr>=0.5

    def sigmoid(self,arr):
        dim = arr.shape
        result = np.ones(dim)
        for i in range(dim[0]):
            for j in range(dim[1]):
                result[i][j] = 1/(1+exp(-arr[i][j])) 
        return result
    
    def feedForward(self):
            #first layer
            #this would compute the !and and or operator
            secondLayerResult = np.matmul(self.secondLayerWeights,self.firstLayer.T) + self.secondLayerBias
            secondLayerResult = self.returnBinary(self.sigmoid(secondLayerResult))
            #second layer, the final answer;
            thirdLayerResult = self.returnBinary(self.sigmoid(np.matmul(secondLayerResult.T,self.thirdLayerWeights.T) + self.thirdLayerBias))
            return thirdLayerResult



m = [[1,1]]
answers = [[0]]
test = XOR(m,answers)
results = test.train()
print(results)


