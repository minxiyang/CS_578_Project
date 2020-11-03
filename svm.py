import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.svm import SVC

class SVM():
    def __init__(self):
        train_data = pd.read_csv("./data/trainingSet.csv")
        test_data = pd.read_csv("./data/testingSet.csv")
        train_data = train_data.sample(frac=1)  # randomly shuffle the train_data

        self.X_train = train_data.drop(['EventId', 'Label'], axis=1)    # train data with col of features
        self.y_train = train_data['Label']                              # train data with col of label
        self.X_test = test_data.drop(['EventId', 'Label'], axis=1)      # test data with col of features
        self.y_test = test_data['Label']                                # test data with col of features

        self.model = None

    def hyperparameter_tune(self):
        max_auc_mean = 0
        best_gamma = None
        best_C = None
        
        n = 5
        for gamma in np.logspace(-2, 2, num=n):
            for C in np.logspace(-2, 2, num=n):
                AUC_mean, AUC_var = self.kfoldcv(gamma, C)
                if (AUC_mean > max_auc_mean):
                    max_auc_mean = AUC_mean
                    best_gamma = gamma
                    best_C = C
        
        return best_gamma, best_C

    def kfoldcv(self, gamma, C):
        k = 5
        n = len(self.X_train)
        d = int(n/k)
        AUCs = []
        for i in range(k):
            train_fold_indices = range(i*d,(i+1)*d)
            test_fold_indices = np.setdiff1d(range(0,n), train_fold_indices)

            self.train(train_fold_indices, gamma, C)
            AUCs.append(self.get_AUC(train_fold_indices, test_fold_indices))
        
        return sum(AUCs)/len(AUCs), np.var(AUCs)

    def train(self, train_fold_indices, gamma, C):
        print("start train")
        X_train_fold = self.X_train.iloc[train_fold_indices]
        y_train_fold = self.y_train.iloc[train_fold_indices]
        
        self.model = SVC(C = C, kernel='rbf', gamma = gamma)
        self.model.fit(X_train_fold, y_train_fold)
        print("finish train")

    def get_AUC(self, train_fold_indices, test_fold_indices):
        print("start get_AUC")
        x_axis = []
        y_axis = []
        X_test_fold = self.X_train.iloc[test_fold_indices]
        y_test_fold = self.y_train.iloc[test_fold_indices]
        decision_func_vals = self.model.decision_function(X_test_fold)
        
        for threshold in np.linspace( min(decision_func_vals), max(decision_func_vals), num=50 ):                
            specificity, sensitivity = self.compute_metrics(y_test_fold, decision_func_vals, threshold)

            x_axis.append(specificity)
            y_axis.append(sensitivity)

            print(specificity, sensitivity)

        print("finish get_AUC")

        #plt.plot(x_axis, y_axis)
        #plt.show()

        return np.trapz(y_axis, x=x_axis)

    def compute_metrics(self, y_test_fold, decision_func_vals, threshold):
        TP = FP = TN = FN = 0
        for i in range(len(y_test_fold)):
            y_actual = y_test_fold.iloc[i]
            y_hat = None
            if decision_func_vals[i] > threshold:
                y_hat = 's'
            else:
                y_hat = 'b'
            
            if y_actual == y_hat == 's':
                TP += 1
            if y_hat == 's' and y_actual != y_hat:
                FP += 1
            if y_actual == y_hat == 'b':
                TN += 1
            if y_hat == 'b' and y_actual != y_hat:
                FN += 1
        
        #accuracy = (TP + TN) / (TP + FP + FN + TN)
        #error = (FP + FN) / (TP + FP + FN + TN)
        #precision = TP / (TP + FP)
        specificity = TN / (TN + FP)
        sensitivity = TP / (TP + FN)
        
        return specificity, sensitivity

if __name__ == "__main__":
    svm = SVM()
    gamma, C = svm.hyperparameter_tune()
    print(svm.kfoldcv(gamma,C))
    print(gamma, C)