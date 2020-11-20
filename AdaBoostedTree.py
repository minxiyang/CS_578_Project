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
					FP+=w.iloc[i]
			elif score[i] < threshold:
				if y_truth=='s':
					FN+=w.iloc[i]
				else:
					TN+=w.iloc[i]
		FPR=np.append(FPR,FP/(TN+FP))
		TPR=np.append(TPR,TP/(TP+FN))
	
	AUC=-np.trapz(TPR,x=FPR)
	print ("AUC is "+str(AUC))
	return [AUC, FPR, TPR]

def find_AMS(FP,TP):
	
	AMS=math.sqrt(2*((FP+TP+10)*math.log(1+TP/(FP+10))-TP))	
	return AMS

def find_threshold(model,x,y,w):

	score=model.decision_function(x)
	AMS=-1
	t=-9999
	TP_t = FP_t = TN_t = FN_t = 0
	for threshold in np.linspace(min(score),max(score),50):

		TP = FP = TN = FN = 0
		for i in range(len(score)):

			y_truth=y.iloc[i]
			if score[i] > threshold:
				if y_truth=='s':
					TP+=w.iloc[i]
				else:
					FP+=w.iloc[i]
			elif score[i] < threshold:
				if y_truth=='s':
                                        FN+=w.iloc[i]
				else:
					TN+=w.iloc[i]
		if (find_AMS(FP,TP)>AMS):
			AMS=find_AMS(FP,TP)
			t=threshold
			TN_t=TN
			TP_t=TP
			FN_t=FN
			FP_t=FP

	print ("threshold is "+str(t))
	print ("AMS is " +str(AMS))
	print ("TN = %s, TP = %s, FN = %s, FP = %s"%(TN_t,TP_t,FN_t,FP_t))
	return t
			


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


	# best: n_est=500 n_lay=4 lr=0.01 AUC=0.9154 AUC_var=6.2064e-06
	# worst: n_est=100 n_lay=6 lr=1 AUC=0.8330 AUC_var=2.0413e-05

	# define model

	best_bdt=AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),n_estimators=500,learning_rate=0.01)
	#worst_bdt=AdaBoostClassifier(DecisionTreeClassifier(max_depth=6),n_estimators=100,learning_rate=1)

	# fit

	best_bdt.fit(x_train,y_train,sample_weight=w_train)
	#worst_bdt.fit(x_train,y_train,sample_weight=w_train)

	# prediction
	[AUC_b, FPR_b, TPR_b]=ROC(best_bdt,x_test,y_test,w_test)
	#[AUC_w, FPR_w, TPR_w]=ROC(worst_bdt,x_test,y_test,w_test)
	
	threshold_b=find_threshold(best_bdt,x_train,y_train,w_train)
	#threshold_w=find_threshold(worst_bdt,x_train,y_train,w_train)

	# find AMS for testing set

	score=best_bdt.decision_function(x_test)
	TP = FP = TN = FN = 0
	for i in range(len(score)):
		y_truth=y_test.iloc[i]
		if score[i] > threshold_b:
			if y_truth=='s':
				TP+=w_test.iloc[i]
			else:
				FP+=w_test.iloc[i]
		elif score[i] < threshold_b:
			if y_truth=='s':
				FN+=w_test.iloc[i]
			else:
				TN+=w_test.iloc[i]
	print ("confusion matrix for testing set")
	print ("TN = %s, TP = %s, FN = %s, FP = %s"%(TN,TP,FN,FP))
	AMS=find_AMS(TN,TP)
	print("AMS = %s"%AMS )	
	# plot result 

	plt.xlabel("FP rate")
	plt.ylabel("TP rate")
	
	plt.plot(FPR_b,TPR_b)
	plt.figtext(0.6,0.9,'AMS = '+str(AMS))
	plt.figtext(0.1,0.9,'AUC = '+str(AUC_b))
	plt.savefig("plots/AdaBoostTree/ROC_final.pdf")
	plt.clf()

	# accuracy versus number of the training sample
	
	test_bdt=AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),n_estimators=500,learning_rate=0.01)
	accuracy=[]
	n_sample=[]	
	for i in range(10):

		sub_idx=range(i*3000,(i+1)*3000)
		train_idx=range((i+1)*3000)
		x_sub=x_train.iloc[sub_idx]
		y_sub=y_train.iloc[sub_idx]
		w_sub=w_train.iloc[sub_idx]
		#test_bdt.fit(x_sub,y_sub,sample_weight=w_sub)
		
		x_train_tot=x_train.iloc[train_idx]
		y_train_tot=y_train.iloc[train_idx]
		w_train_tot=w_train.iloc[train_idx]
		test_bdt.fit(x_train_tot,y_train_tot,sample_weight=w_train_tot)
		threshold=find_threshold(test_bdt,x_train_tot,y_train_tot,w_train_tot)
		score=test_bdt.decision_function(x_test)
		TP = FP = TN = FN = 0
		
		for j in range(len(score)):
			y_truth=y_test.iloc[j]
			if score[j] > threshold:
				if y_truth=='s':
					TP+=w_test.iloc[j]
				else:
					FP+=w_test.iloc[j]
			elif score[j] < threshold:
				if y_truth=='s':
					FN+=w_test.iloc[j]
				else:
					TN+=w_test.iloc[j]	
		acc=(TP+TN)/(TP+FP+TN+FN)
		accuracy.append(acc)
		n_sample.append((i+1)*3000)
		print ("for step %s, the accuracy is %s"%(str(i),str(acc)))	
	plt.xlabel("number of samples")
	plt.ylabel("accuracy")
	plt.plot(n_sample,accuracy)
	plt.savefig("plots/AdaBoostTree/accVsnSamples.pdf")
	plt.clf()		
