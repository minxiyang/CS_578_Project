import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.svm import SVC
import sys

class SVM():
    def __init__(self):
        train_data = pd.read_csv("./data/trainingSet.csv")
        test_data = pd.read_csv("./data/testingSet.csv")
        train_data = train_data.sample(frac=1)  # randomly shuffle the train_data

        self.X_train = train_data.drop(['EventId', 'Weight', 'Label'], axis=1)      # train data with col of features
        self.y_train = train_data['Label']                                          # train data with col of label
        self.w_train = train_data['Weight']

        self.X_test = test_data.drop(['EventId', 'Weight', 'Label'], axis=1)        # test data with col of features
        self.y_test = test_data['Label']                                            # test data with col of features
        self.w_test = test_data['Weight']

        self.model = None

    def hyperparameter_tune(self):
        max_auc_mean = 0
        best_gamma = None
        best_C = None

        n = 5
        arr = []
        for gamma in np.logspace(-2, 2, num=n):
            a = []
            for C in np.logspace(-2, 2, num=n):
                AUC_mean, AUC_var = self.kfoldcv(gamma, C, False)
                if (AUC_mean > max_auc_mean and AUC_var < 1):
                    max_auc_mean = AUC_mean
                    best_gamma = gamma
                    best_C = C
                a.append('(' + str(gamma) + ', ' + str(C) + ')' + str(round(AUC_mean, 4)) + ' ' + str(round(AUC_var, 4)) + ' ')
            arr.append(a)
        
        print()
        print("Mean and var of AUCs in hyperparam tuning")
        print(arr)
        print()
        
        return best_gamma, best_C

    def kfoldcv(self, gamma, C, draw_ROC):
        k = 5
        n = len(self.X_train)
        d = int(n/k)
        AUCs = k*[None]

        for i in range(k):
            test_fold_indices = range(i*d,(i+1)*d)
            train_fold_indices = np.setdiff1d(range(0,n), test_fold_indices)

            X_train_fold    = self.X_train.iloc[train_fold_indices]
            y_train_fold    = self.y_train.iloc[train_fold_indices]
            w_train_fold    = self.w_train.iloc[train_fold_indices]
            X_test_fold     = self.X_train.iloc[test_fold_indices]
            y_test_fold     = self.y_train.iloc[test_fold_indices]
            w_test_fold     = self.w_train.iloc[test_fold_indices]
            
            self.train(X_train_fold, y_train_fold, w_train_fold, gamma, C)

            decision_func_vals = self.model.decision_function(X_test_fold)
            x_axis, y_axis = self.get_ROC_curve(decision_func_vals, y_test_fold, w_train_fold)
            AUC = np.trapz(y_axis, x=x_axis)
            AUCs[i] = AUC
        
        return sum(AUCs)/len(AUCs), np.var(AUCs)

    def train(self, X, y, weight, gamma, C):
        print("start train")

        self.model = SVC(C = C, kernel='rbf', gamma = gamma)
        self.model.fit(X, y, sample_weight = weight)

        print("finish train")

    def get_ROC_curve(self, decision_func_vals, y, w):
        print("start get_ROC_curve")

        num = 50
        x_ROC = num*[None]
        y_ROC = num*[None]

        i = 0        
        for threshold in np.linspace( min(decision_func_vals), max(decision_func_vals), num=num ):                
            TP, TN, FP, FN = self.compute_TP_TN_FP_FN(y, w, decision_func_vals, threshold)
            specificity, sensitivity = self.compute_spec_sens(TP, TN, FP, FN)

            x_ROC[i] = specificity
            y_ROC[i] = sensitivity
            i += 1

        print("finish get_ROC_curve")

        return x_ROC, y_ROC

    def compute_TP_TN_FP_FN(self, y, w, decision_func_vals, threshold):
        TP = FP = TN = FN = 0
        for i in range(len(y)):
            y_actual = y.iloc[i]
            y_hat = None
            if decision_func_vals[i] > threshold:
                y_hat = 's'
            else:
                y_hat = 'b'
            
            if y_actual == y_hat == 's':
                TP += w.iloc[i]
            if y_hat == 's' and y_actual != y_hat:
                FP += w.iloc[i]
            if y_actual == y_hat == 'b':
                TN += w.iloc[i]
            if y_hat == 'b' and y_actual != y_hat:
                FN += w.iloc[i]
        
        return TP, TN, FP, FN
    
    def compute_spec_sens(self, TP, TN, FP, FN):
        specificity = TN / (TN + FP)
        sensitivity = TP / (TP + FN) 

        return specificity, sensitivity

    def test(self, gamma, C):
        self.train(self.X_train, self.y_train, self.w_train, gamma, C)
        scores_on_test = self.model.decision_function(self.X_test)
        scores_on_train = self.model.decision_function(self.X_train)

        x_ROC, y_ROC        = self.get_ROC_curve(scores_on_test, self.y_test, self.w_test)
        AUC                 = np.trapz(y_ROC, x=x_ROC)
        best_thres          = self.get_best_AMS_thres(scores_on_train, self.y_train, self.w_train)
        TP, TN, FP, FN      = self.compute_TP_TN_FP_FN(self.y_test, self.w_test, scores_on_test, best_thres)
        max_ams             = self.AMS(TP, FP)

        self.plot_curve(x_ROC, y_ROC, 'Specificity', 'Sensitivity', 'ROC', 'SVM_ROC.png', gamma, C, AUC)
        print()
        print("Best threshold that maximizes AMS")
        print("Maximum AMS = " + str(round(max_ams, 4)) + " with threshold = " + str(round(best_thres, 4)))
        print()
        print("Confusion matrix with best threshold")
        print("TP = " + str(round(TP, 4)) + ", FP = " + str(round(FP, 4)))
        print("FN = " + str(round(FN, 4)) + ", TN = " + str(round(TN, 4)))
        print()
        self.accuracy_VS_numTrain(gamma, C)

    def accuracy_VS_numTrain(self, gamma, C):
        k = 30
        n = len(self.X_train)
        d = int(n/k)
        x_axis = [None]*k
        y_axis = [None]*k

        for i in range(k):
            train_fold_indices = range(i*d,(i+1)*d)

            X_train_fold    = self.X_train.iloc[train_fold_indices]
            y_train_fold    = self.y_train.iloc[train_fold_indices]
            w_train_fold    = self.w_train.iloc[train_fold_indices]
            
            self.train(X_train_fold, y_train_fold, w_train_fold, gamma, C)

            scores_on_train = self.model.decision_function(X_train_fold)
            scores_on_test  = self.model.decision_function(self.X_test)

            best_thres      = self.get_best_AMS_thres(scores_on_train, y_train_fold, w_train_fold)
            TP, TN, FP, FN  = self.compute_TP_TN_FP_FN(self.y_test, self.w_test, scores_on_test, best_thres)
            accuracy        = (TP + TN) / (TP + FP + FN + TN)

            x_axis[i]       = (i+1)*1000
            y_axis[i]       = accuracy
        
        x_label  = 'Number of Training Sample'
        y_label  = 'Accuracy'
        title    = 'Accur vs. num of train sample'
        filename = 'SVM_accuracies.png'

        self.plot_curve(x_axis, y_axis, x_label, y_label, title, filename, gamma, C, -1)

    def get_best_AMS_thres(self, decision_func_vals, y, w):
        max_ams = 0
        best_thres = 0

        num = 50
        for threshold in np.linspace( min(decision_func_vals), max(decision_func_vals), num=num ):                
            TP, TN, FP, FN = self.compute_TP_TN_FP_FN(y, w, decision_func_vals, threshold)
            ams = self.AMS(TP, FP)

            if (ams > max_ams):
                max_ams = ams
                best_thres = threshold

        return best_thres 
    
    def plot_curve(self, x_axis, y_axis, x_label, y_label, title, filename, gamma, C, AUC):
        plt.figure()
        plt.xlabel(x_label)
        plt.ylabel(y_label) 
        plt.title(title + ' of SVM with gamma = ' + str(gamma) + ', C = ' + str(C))
        if AUC > 0:
            plt.plot(x_axis, y_axis, label="AUC = " + str(round(AUC,4)))
            plt.legend()
        else:
            plt.plot(x_axis, y_axis)
        plt.savefig('./plots/SVM/' + filename)

    def AMS(self, TP, FP):
        s = TP
        b = FP
        return math.sqrt( 2*( (s + b + 10)*math.log(1 + s/(b+10)) - s ) )

if __name__ == "__main__":
    svm = SVM()
    if int(sys.argv[1]) == 1:
        gamma, C = 0.01, 100
    else:
        gamma, C = svm.hyperparameter_tune()
    print()
    print("Best hyperparameters")
    print("Gamma = " + str(gamma) + ", C = " + str(C))
    print()
    svm.test(gamma, C)