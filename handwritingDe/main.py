from numpy import *


def imageToVector(filename):
    returnVec = zeros((1,1024))
    with (open(filename)) as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVec[0,32*i + j] = int(lineStr[j])
    return returnVec

def handwritingClassTest():
    hwLabels = []
    trainingFileList = ['test.txt','test2.txt','test3.txt','test4.txt']   
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = imageToVector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = imageToVector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
             errorCount += 1.0
        print ("\nthe total number of errors is: %d" % errorCount)
        print ("\nthe total error rate is: %f" % (errorCount/float(mTest))