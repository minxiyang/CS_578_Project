import matplotlib.pyplot as plt
import pandas as pd
import numpy as np




def plotFeatures(data,path,sigScale=1,isLog=False):
	for label, content in data.items():
		if label!="EventId" and label!="Weight" and label!="Label":
			SigWeight=data.loc[(data['Label']=="s"),["Weight"]]
			SigWeight=SigWeight.astype(float)
			SigWeight=SigWeight.to_numpy()
			SigWeight=SigWeight.reshape(-1,)
			SigWeight=SigWeight*sigScale
			SigFeature=data.loc[data['Label']=="s",[label]]
			SigFeature=SigFeature.astype(float)
			SigFeature=SigFeature.to_numpy()
			SigFeature=SigFeature.reshape(-1,)
			plt.hist(SigFeature,100,weights=SigWeight,label="signal",alpha=0.5,facecolor="red")

			BkgWeight=data.loc[data['Label']=="b",["Weight"]]
			BkgWeight=BkgWeight.astype(float)
			BkgWeight=BkgWeight.to_numpy()
			BkgWeight=BkgWeight.reshape(-1,)
			BkgFeature=data.loc[data['Label']=="b",[label]]
			BkgFeature=BkgFeature.astype(float)
			BkgFeature=BkgFeature.to_numpy()
			BkgFeature=BkgFeature.reshape(-1,)

			plt.hist(BkgFeature,100,weights=BkgWeight,label="background",alpha=0.5,facecolor="green")
			plt.legend(loc="upper right")
			plt.xlabel(label)
			plt.ylabel("distribution")
			if isLog:
				plt.yscale("log")
			plt.savefig(path+label+".pdf")
			plt.clf()
						


