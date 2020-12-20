import numpy as np
from math import exp
class Storage():
    def __init__(self,first,second,answer):
        self.first = first
        self.second = second
        self.answer = answer


def sigmoid(arr):
    dim  = arr.shape
    result = np.ones(dim)
    for i in range(dim[0]):
        for j in range(dim[1]):
            result[i][j] = 1/(1+exp(-arr[i][j]))

    return result
def generateRandom(row,col):
    arr = []
    for i in range(row):
        temp=[]
        for j in range(col):
            temp.append(2*np.random.rand() - 1)
        arr.append(temp)
    return np.array(arr)

class NeuralNetwork():

    def __init__(self,inputs):
        self.inputs  = np.array(inputs)


        # self.secondLayerWeights = np.random.random_sample(size=(2,2))
        # self.thirdLayerWeights = np.random.random_sample(size=(1,2))
        # self.secondLayerBias = np.random.random_sample(size=(2,1))
        # self.thirdLayerBias = np.random.random_sample(size=(1,1))
        self.secondLayerWeights = generateRandom(2,2)
        self.thirdLayerWeights = generateRandom(1,2)
        self.secondLayerBias = generateRandom(2,1)
        self.thirdLayerBias = generateRandom(1,1)
        


        self.alpha = 0.1


    def feedForward(self,input):
        secondLayerResult = (np.matmul(self.secondLayerWeights,input))
        secondLayerResult += self.secondLayerBias
        secondLayerResult = sigmoid(secondLayerResult)
        result = np.matmul(self.thirdLayerWeights,secondLayerResult)
        result += self.thirdLayerBias
        result = sigmoid(result)
        return result

    def train(self,input,answer):
        hidden = (np.matmul(self.secondLayerWeights,input)) +self.secondLayerBias
        hidden = sigmoid(hidden)

        outputs = np.matmul(self.thirdLayerWeights,hidden) + self.thirdLayerBias
        outputs = sigmoid(outputs)
        #gradient for the third layer.
        outputError = answer - outputs
        temp =self.alpha*(outputs*(1-outputs))*outputError

        gradient = np.matmul(temp,hidden.T)
        self.thirdLayerWeights += gradient
        self.thirdLayerBias += temp

        #gradient for the second layer.
        secondLayerHiddenError = np.matmul(self.thirdLayerWeights.T,outputError)
        temp =self.alpha*secondLayerHiddenError*(hidden*(1-hidden))
        hiddenGradient = np.matmul(temp,input.T)
        self.secondLayerWeights += hiddenGradient
        self.secondLayerBias += temp

        return
    def run(self):

            inp  = np.random.choice(self.inputs)

         
            (self.train(np.array([[inp.first],[inp.second]]),np.array([[inp.answer]])))
# inputs = [[[1,0]],[[1,1]],[[0,0]],[[0,1]]]
# answers = [[1],[0],[0],[1]]
inputs = [Storage(1,0,1),Storage(1,1,1),Storage(0,0,0),Storage(0,1,1)]
nn = NeuralNetwork(inputs)
for i in range(100000):
    nn.run()
print(nn.feedForward(np.array([[1],[0]])))
print(nn.feedForward(np.array([[0],[1]])))
print(nn.feedForward(np.array([[1],[1]])))
print(nn.feedForward(np.array([[0],[0]])))
