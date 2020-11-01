import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import log

def subset(data,nTrain=30000,nTest=20000): # select 50000 events and divide into 30000 training and 20000 testing

        dataS=data.loc[data["Label"]=="s"]
        n=round((nTrain+nTest)/2)
        subDataS=dataS.sample(n)
        dataTrainS=subDataS.iloc[:round(nTrain/2)]
        dataTestS=subDataS.iloc[round(nTrain/2):]
        dataB=data.loc[data["Label"]=="b"]
        subDataB=dataB.sample(n)
        dataTrainB=subDataB.iloc[:round(nTrain/2)]
        dataTestB=subDataB.iloc[round(nTrain/2):]
        trainingData=pd.concat([dataTrainS,dataTrainB])
        testingData=pd.concat([dataTestS,dataTestB])
        return [trainingData,testingData]
