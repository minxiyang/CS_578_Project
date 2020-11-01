import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import log

def featureScaling(data,trainingSet=False): # preprocessing the data
	
	if trainingSet:
		sigMean=data.loc[data["Label"]=="s",["Weight"]].mean()
		data.loc[data["Label"]=="s",["Weight"]]=data.loc[data["Label"]=="s",["Weight"]]/sigMean
		bkgMean=data.loc[data["Label"]=="b",["Weight"]].mean()
		data.loc[data["Label"]=="b",["Weight"]]=data.loc[data["Label"]=="b",["Weight"]]/bkgMean


	logScale={"DER_pt_tot":500.,"PRI_jet_leading_pt":800.,"PRI_jet_subleading_pt":500.,"PRI_lep_pt":450.,"PRI_met":750.,"PRI_met_sumet":1500.,"PRI_tau_pt":400.,"DER_pt_ratio_lep_tau":16.,"DER_pt_h":770.,"DER_mass_vis":800.,"DER_mass_transverse_met_lep":520.,"DER_mass_jet_jet":4000.,"DER_mass_MMC":1000.,"PRI_jet_all_pt":1300.,"DER_sum_pt":1400.}

	
	for label in logScale.keys():

		data.loc[(data[label]>logScale[label]),[label]]=logScale[label]
		data.loc[(data[label]>-999.),[label]]=data.loc[(data[label]>-999.),[label]].apply(lambda x:log(x+1))	

	for label, content in data.items():
		if label!="EventId" and label!="Weight" and label!="Label" and label!="PRI_jet_num":

			std=data.loc[(data[label]>-999.),[label]].std()
			mean=data.loc[(data[label]>-999.),[label]].mean()
			data.loc[(data[label]>-999.),[label]]=(data.loc[data[label]>-999.,[label]]-mean)/std
			minimal=data.loc[(data[label]>-999.),[label]].min()
			minimal=minimal.values
			data.loc[(data[label]==-999.),[label]]=minimal-1
	


