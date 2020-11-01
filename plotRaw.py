import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data=pd.read_csv("data/training.csv")
#print(data.head())
#print(data.describe())
#print(data.columns)
for label, content in data.items():
	if label!="EventId" and label!="Weight" and label!="Label":
		SigWeight=data.loc[(data[label]>-999) & (data['Label']=="s"),["Weight"]]
		SigWeight=SigWeight.astype(float)
		SigWeight=SigWeight.to_numpy()
		SigWeight=SigWeight.reshape(-1,)
		SigWeightScaled=SigWeight*100
		#print(len(SigWeight))
		SigFeature=data.loc[(data[label]>-999) & (data['Label']=="s"),[label]]
		SigFeature=SigFeature.astype(float)
		SigFeature=SigFeature.to_numpy()
		SigFeature=SigFeature.reshape(-1,)
		plt.hist(SigFeature,100,weights=SigWeightScaled,label="signal*100",alpha=0.5,facecolor="red")

		BkgWeight=data.loc[(data[label]>-999) & (data['Label']=="b"),["Weight"]]
		BkgWeight=BkgWeight.astype(float)
		BkgWeight=BkgWeight.to_numpy()
		BkgWeight=BkgWeight.reshape(-1,)
		#print(len(BkgWeight))
		BkgFeature=data.loc[(data[label]>-999) & (data['Label']=="b"),[label]]
		BkgFeature=BkgFeature.astype(float)
		BkgFeature=BkgFeature.to_numpy()
		BkgFeature=BkgFeature.reshape(-1,)

		#plt.hist([SigFeature,BkgFeature],100,weights=[SigWeightScaled,BkgWeight],label=["signal*100","background"],alpha=0.5)#,facecolor=["red","green"])
		plt.hist(BkgFeature,100,weights=BkgWeight,label="background",alpha=0.5,facecolor="green")
		plt.legend(loc="upper right")
		plt.xlabel(label)
		plt.ylabel("distribution")
		plt.savefig("plots/plots_raw/"+label+".pdf")
		plt.yscale("log")
		plt.savefig("plots/plots_raw_log/"+label+".pdf")
		plt.clf()
						
SigWeight=data.loc[data['Label']=="s",["Weight"]]
SigWeight=SigWeight.astype(float)
SigWeight=SigWeight.to_numpy()
SigWeight=SigWeight.reshape(-1,)
plt.hist(SigWeight,100,label="signal",alpha=0.5,facecolor="red")
plt.xlabel("signal weight")
plt.ylabel("distribution")
plt.savefig("plots/plots_raw/signalWeight.pdf")
plt.clf()

BkgWeight=data.loc[data['Label']=="b",["Weight"]]                       
BkgWeight=BkgWeight.astype(float)
BkgWeight=BkgWeight.to_numpy()
BkgWeight=BkgWeight.reshape(-1,)
plt.hist(BkgWeight,100,label="background",alpha=0.5,facecolor="green")
plt.xlabel("background weight")
plt.ylabel("distribution")
plt.savefig("plots/plots_raw/backgroundWeight.pdf")
plt.clf()

SigWeight=data.loc[data['Label']=="s",["Weight"]]	


