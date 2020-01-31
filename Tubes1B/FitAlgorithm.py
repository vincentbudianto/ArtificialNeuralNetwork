# Main program to fit the data read

import pandas as pd
import numpy as np
import Function as f
import DecisionTree

# Get data from csv
def getCSVData(fileName):
    data = pd.read_csv(fileName)
    dataHead = list(data.columns)
    data = np.array(data)
    return (dataHead, data)

# Data splitter
def splitXY(data):
    newX = []
    newY = []
    fullSize =


# Parsing ke bentuk number
def numberParser(data):
    print("test")


# Create a basic fitying algorithn
def fit(data):
    decisionTree = DecisionTree()





# print(getCSVData("tennis.csv"))