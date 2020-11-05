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

        self.X_train = train_data.drop(['EventId', 'Weight', 'Label'], axis=1)      # train data with col of features
        self.X_train_weight = train_data['Weight']
        self.y_train = train_data['Label']                                          # train data with col of label

        self.X_test = test_data.drop(['EventId', 'Weight', 'Label'], axis=1)        # test data with col of features
        self.X_test_weight = test_data['Weight']
        self.y_test = test_data['Label']                                            # test data with col of features

        self.model = None

    def hyperparameter_tune(self):
        max_auc_mean = 0
        best_gamma = None
        best_C = None
        
        n = 5
        for gamma in np.logspace(-2, 2, num=n):
            for C in np.logspace(-2, 2, num=n):
                AUC_mean, AUC_var = self.kfoldcv(gamma, C, False)
                if (AUC_mean > max_auc_mean and AUC_var < 3):
                    max_auc_mean = AUC_mean
                    best_gamma = gamma
                    best_C = C
        
        return best_gamma, best_C

    def kfoldcv(self, gamma, C, draw_ROC):
        k = 5
        n = len(self.X_train)
        d = int(n/k)
        AUCs = k*[None]

        for i in range(k):
            train_fold_indices = range(i*d,(i+1)*d)
            test_fold_indices = np.setdiff1d(range(0,n), train_fold_indices)

            X_train_fold = self.X_train.iloc[train_fold_indices]
            X_weight_fold = self.X_train_weight.iloc[train_fold_indices]
            y_train_fold = self.y_train.iloc[train_fold_indices]
            X_test_fold = self.X_train.iloc[test_fold_indices]
            y_test_fold = self.y_train.iloc[test_fold_indices]
            
            self.train(X_train_fold, y_train_fold, X_weight_fold, gamma, C)

            x_axis, y_axis = self.get_ROC_curve(X_test_fold, y_test_fold)
            AUC = np.trapz(y_axis, x=x_axis)
            AUCs[i] = AUC
        
        return sum(AUCs)/len(AUCs), np.var(AUCs)

    def train(self, X, y, weight, gamma, C):
        print("start train")

        self.model = SVC(C = C, kernel='rbf', gamma = gamma)
        self.model.fit(X, y, sample_weight = weight)

        print("finish train")

    def get_ROC_curve(self, X, y):
        print("start get_ROC_curve")

        num = 50
        x_axis = num*[None]
        y_axis = num*[None]
        decision_func_vals = self.model.decision_function(X)

        i = 0        
        for threshold in np.linspace( min(decision_func_vals), max(decision_func_vals), num=num ):                
            specificity, sensitivity = self.compute_metrics(y, decision_func_vals, threshold)

            x_axis[i] = specificity
            y_axis[i] = sensitivity
            i += 1

        print("finish get_ROC_curve")

        return x_axis, y_axis

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

    def test(self, gamma, C):
        self.train(self.X_train, self.y_train, self.X_train_weight, gamma, C)

        x_axis, y_axis = self.get_ROC_curve(self.X_test, self.y_test)
        AUC = np.trapz(y_axis, x=x_axis)

        plt.xlabel('Specificity')
        plt.ylabel('Sensitivity') 
        plt.title('SVM ROC with gamma = ' + str(gamma) + ', C = ' + str(C))
        plt.plot(x_axis, y_axis, label="AUC = " + str(round(AUC,4)))
        plt.legend()
        plt.savefig('SVM_ROC.png')

        return AUC

if __name__ == "__main__":
    svm = SVM()
    gamma, C = svm.hyperparameter_tune()
    print(svm.test(gamma, C))           # draw ROC and compute AUC with best parameters
    print(gamma, C)                     # the best parameters chosen from hyperparameter tuning