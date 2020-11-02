import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from subset import subset
from featureScaling import featureScaling
from plotAllFeature import plotFeatures





data=pd.read_csv("data/training.csv")
plotPath="plots/plots_preprocessing/"
[trainingSet,testingSet]=subset(data)
featureScaling(trainingSet,True)
featureScaling(testingSet)
plotFeatures(trainingSet,plotPath)

trainingSet.to_csv("data/trainingSet.csv",index=False,header=True)
testingSet.to_csv("data/testingSet.csv",index=False,header=True)
