from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.base import clone
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


def ROC(model,x,y,w):
	
	FPR=np.ones(0)
	TPR=np.ones(0)
	score=model.decision_function(x)
	for threshold in np.linspace(min(score),max(score),50):

		TP = FP = TN = FN = 0
		for i in range(len(score)):
			
			y_truth=y.iloc[i]
			if score[i] > threshold:
				if y_truth=='s':
					TP+=w.iloc[i]
				else:
					TN+=w.iloc[i]
			elif score[i] < threshold:
				if y_truth=='s':
					FN+=w.iloc[i]
				else:
					FP+=w.iloc[i]
		FPR=np.append(FPR,TN/(TN+FP))
		TPR=np.append(TPR,TP/(TP+FN))
	
	AUC=-np.trapz(TPR,x=FPR)
	print ("AUC is "+str(AUC))
	return [AUC, FPR, TPR]
	
			

# k-fold cross vaildation
def kfold(k,bdt,x,y,w):

	n=len(x)
	d=round(n/k)
	k_fold_idx=[]

	print ('start k-fold cross vaildation')
	for i in range(k):
	
		test_idx=range(d*i,d*(i+1))
		train_idx=np.setdiff1d(range(0,n),test_idx)
		k_fold_idx.append([train_idx,test_idx])

	AUCs=[]
	FPRs=[]
	TPRs=[]

	for i in range(k):
	
		print ('run %s time k-fold' %str(i+1))
		[train_idx,test_idx]=k_fold_idx[i]
		x_fold=x.iloc[train_idx]
		y_fold=y.iloc[train_idx]
		w_fold=w[train_idx]
		x_test_fold=x.iloc[test_idx]
		y_test_fold=y.iloc[test_idx]
		w_test_fold=w.iloc[test_idx]
		model=clone(bdt)
		model.fit(x_fold,y_fold,sample_weight=w_fold)
		[AUC,FPR,TPR]=ROC(model,x_test_fold,y_test_fold,w_test_fold)
		AUCs.append(AUC)
		FPRs.append(FPR)
		TPRs.append(TPR)

	print ("k-fold result")

	AUC_var=np.var(AUCs)
	AUC_mean=sum(AUCs)/len(AUCs)

	print('variance for AUC is '+str(AUC_var))
	print('mean for AUC is '+str(AUC_mean))
	return [AUC_mean,AUC_var,FPRs,TPRs]

if __name__=="__main__":
	
	#load data                              

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

	#define model 
	for n_est in [1,3,5]:
		for n_lay in [3,4,5]:
			for lr in [0.01]:

				bdt=AdaBoostClassifier(DecisionTreeClassifier(max_depth=n_lay),n_estimators=n_est,learning_rate=lr)
	
	# k-fold cross vaildation

				k=5
	
				[AUC_mean,AUC_var,FPRs,TPRs]=kfold(k,bdt,x_train,y_train,w_train)
	#plot ROC

				plt.xlabel("FP rate")
				plt.ylabel("TP rate")
	
				for i in range(k):

					plt.plot(FPRs[i],TPRs[i])

				plt.figtext(0.1,0.9,'AUC_mean='+str(AUC_mean))
				plt.figtext(0.6,0.9,'AUC_var='+str(AUC_var))
				plt.savefig("plots/AdaBoostTree/ROC_kfold_nest"+str(n_est)+"_lay"+str(n_lay)+"_lr"+str(lr)+".pdf")
				plt.clf()

