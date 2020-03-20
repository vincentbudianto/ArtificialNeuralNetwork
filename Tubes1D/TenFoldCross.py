# External library
import pandas as pd
import numpy as np
import copy as cp
import math
import time
import pickle

# Internal library
from decision_tree.FitC45 import prune, createRule, getErrorCountExternal
from multilayer_perceptron.main import learn, nodeOutputCheckExternal

'''
Class TenFoldCross
Purpose read dataset to be learnt by ANN and DTL
Show the result
'''
class TenFoldCross:
    '''
    Constructor
    '''
    def __init__(self, fileNameValue):
        # Get filename
        fileName = "dataset/" + fileNameValue

        # Get shuffled data
        self.data = pd.read_csv(fileName).sample(frac=1).reset_index(drop=True)
        self.dataLength = len(self.data)
        self.dataHead = list(self.data.columns)
        self.modelDTL = None
        self.modelMLP = None

        dataCopy = cp.copy(self.data)
        self.dataNPArray = np.array(dataCopy)

    '''
    One time run
    '''
    def runDTL(self, data):
        dataCopy = cp.copy(data)
        dataNPArray = np.array(dataCopy)
        return prune(self.dataHead, dataNPArray, data, None, None)

    def runANN(self, data):
        predictData = data
        return learn(data, self.dataHead, predictData)


    '''
    Get dataframe needed for doing Ten Fold Cross Validation
    '''
    def gatheringData(self):
        # Result
        trainingResult = []
        testingResult = []

        # Get splitted size for every iteration
        splitSize = math.ceil(self.dataLength / 10)
        tempArray = np.array(self.data)

        # Now split the data
        for i in range(10):
            # Get data for every loop (already randomized at init)
            tempTrainingData = []
            tempTestingData = []
            for j in range(len(tempArray)):
                if (j >= i * splitSize) and (j < (i + 1) * splitSize):
                    tempTestingData.append((tempArray[j][0], tempArray[j][1], tempArray[j][2], tempArray[j][3], tempArray[j][4]))
                else:
                    tempTrainingData.append((tempArray[j][0], tempArray[j][1], tempArray[j][2], tempArray[j][3], tempArray[j][4]))

            # Convert data to datagram
            trainingData = pd.DataFrame(tempTrainingData, columns=['sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'variety'])
            testingData = pd.DataFrame(tempTestingData, columns=['sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'variety'])

            # Getting data for every
            trainingResult.append(trainingData)
            testingResult.append(testingData)

        return (trainingResult, testingResult)

    '''
    Doing the ten cross fold validation
    '''
    def tenCrossFoldDTL(self):
        trainingTimeList = []
        accuracyList = []

        # Get training and testing data
        trainingData, testingData = self.gatheringData()

        # Loop for each data
        for i in range(10):
            # Train the data
            timeStart = time.time()
            model = self.runDTL(trainingData[i])

            modelRules = []
            for i in range(len(model)):
                modelRules.append(createRule(model[i]))
            timeEnd = time.time()

            trainingTimeList.append(timeEnd - timeStart)

            # Check for data's accuracy
            errorCount = getErrorCountExternal(testingData[i], modelRules, self.dataHead)
            accuracy = (len(testingData[i]) - errorCount) / len(testingData[i])
            accuracyList.append(accuracy)

        print("Elapsed Time")
        print(trainingTimeList)

        print("Accuracy")
        print(accuracyList)


    '''
    Doing the ten cross fold validation
    '''
    def tenCrossFoldANN(self):
        trainingTimeList = []
        accuracyList = []

        # Get training and testing data
        trainingData, testingData = self.gatheringData()

        model = None
        try:
            file = open('save_file_ann', 'rb')
            decision = input("ANN model save_file found, do you want to load model (y/n)?")
            if decision[0].lower() == 'y':
                model = pickle.load(file)
                print('ANN model is loaded from file successfully!')
            else:
                print('ANN model is not loaded from file!')
            file.close()
        except IOError:
            print('ANN model save_file not found, initiating new model!')


        # Loop for each data
        for i in range(10):
            # Train the data
            timeStart = time.time()
            model = self.runANN(trainingData[i])

            file = open('save_file_ann', 'wb')
            pickle.dump(model, file)
            file.close()

            print('Saving newest ANN model!')
            timeEnd = time.time()

            trainingTimeList.append(timeEnd - timeStart)

            # Check for data's accuracy
            accuracy = model.predict(testingData[i], nodeOutputCheckExternal)
            accuracyList.append(accuracy)

        print("Elapsed Time")
        print(trainingTimeList)

        print("Accuracy")
        print(accuracyList)


# Main function

# Testing DTL
# tenfold = TenFoldCross("iris.csv")
# tenfold.runDTL(tenfold.data, tenfold.dataNPArray)

# Testing ANN
# tenfold = TenFoldCross("iris.csv")
# tenfold.runANN(tenfold.data)

# Testing executiong
# tenfold = TenFoldCross("iris.csv")
# tenfold.tenCrossFoldDTL()

# Testing executiong
tenfold = TenFoldCross("iris.csv")
tenfold.tenCrossFoldANN()