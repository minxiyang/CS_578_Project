from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.base import clone
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from AdaBoostedTree import *





if __name__=="__main__":

	train_data = pd.read_csv("./data/trainingSet.csv")
	train_data.loc[train_data['Label']=='s',['Weight']]= train_data.loc[train_data['Label']=='s',['Weight']]/123.02614
	train_data = train_data.sample(frac=1)
	test_data = pd.read_csv("./data/testingSet.csv")
	x_train = train_data.drop(['EventId','Weight','Label'], axis=1)
	w_train = train_data['Weight']
	y_train = train_data['Label']
	x_test = test_data.drop(['EventId','Weight','Label'], axis=1)
	w_test = test_data['Weight']
	y_test = test_data['Label']
	for n_est in [1,3,5,10,100,500]:
		for n_lay in [3,4,5,6,7]:
			for lr in [0.01,0.1,1]:

				bdt=AdaBoostClassifier(DecisionTreeClassifier(max_depth=n_lay),n_estimators=n_est,learning_rate=lr)
				k=5
				[AUC_mean,AUC_var,FPRs,TPRs]=kfold(k,bdt,x_train,y_train,w_train)
				plt.xlabel("FP rate")
				plt.ylabel("TP rate")
				for i in range(k):

					plt.plot(FPRs[i],TPRs[i])

				plt.figtext(0.1,0.9,'AUC_mean='+str(AUC_mean))
				plt.figtext(0.6,0.9,'AUC_var='+str(AUC_var))
				plt.savefig("plots/AdaBoostTree/ROC_kfold_nest"+str(n_est)+"_lay"+str(n_lay)+"_lr"+str(lr)+".pdf")
				plt.clf()
