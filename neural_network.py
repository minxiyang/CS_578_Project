
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import errno

import pandas as pd




# Basic Setting
parser = argparse.ArgumentParser(description='Higgs Boson')
parser.add_argument('--seed', default=1, type = int, help = 'set seed')

# Training Setting
parser.add_argument('--nepoch', default = 300, type = int, help = 'total number of training epochs')
parser.add_argument('--lr', default = 0.001, type = float, help = 'initial learning rate')
parser.add_argument('--momentum', default = 0.9, type = float, help = 'momentum in SGD')
parser.add_argument('--weight_decay', default = 0.0001, type = float, help = 'weight decay in SGD')
parser.add_argument('--batch_train', default = 128, type = int, help = 'batch size for training')

parser.add_argument('--n_hidden', default=2, type = str, help = 'number of hidden layer')
parser.add_argument('--activation', default='relu', type = str, help = 'set activation function')


args = parser.parse_args()

class Net_tanh(nn.Module):
    def __init__(self, input_dim, num_hidden, output_dim):
        super(Net_tanh, self).__init__()
        self.num_hidden = num_hidden
        self.fc_list = []
        self.fc_list.append(nn.Linear(input_dim, 30))
        self.add_module('fc1', self.fc_list[-1])
        for i in range(num_hidden - 1):
            self.fc_list.append(nn.Linear(30, 30))
            self.add_module('fc'+str(i+2), self.fc_list[-1])
        self.fc_list.append(nn.Linear(30, output_dim))
        self.add_module('fc' + str(num_hidden + 1), self.fc_list[-1])

    def forward(self, x):
        for i in range(self.num_hidden):
            x = torch.tanh(self.fc_list[i](x))
        x = self.fc_list[-1](x)
        return x


class Net_relu(nn.Module):
    def __init__(self, input_dim, num_hidden, output_dim):
        super(Net_relu, self).__init__()
        self.num_hidden = num_hidden
        self.fc_list = []
        self.fc_list.append(nn.Linear(input_dim, 30))
        self.add_module('fc1', self.fc_list[-1])
        for i in range(num_hidden - 1):
            self.fc_list.append(nn.Linear(30, 30))
            self.add_module('fc'+str(i+2), self.fc_list[-1])
        self.fc_list.append(nn.Linear(30, output_dim))
        self.add_module('fc' + str(num_hidden + 1), self.fc_list[-1])

    def forward(self, x):
        for i in range(self.num_hidden ):
            x = torch.relu(self.fc_list[i](x))
        x = self.fc_list[-1](x)
        return x


def cal_auc(output, target, num_bin):
    min_val = output.min().item()
    max_val = output.max().item()
    threshold = np.linspace(min_val, max_val, num_bin)
    sensitivity_list = np.zeros([num_bin])
    specificity_list = np.zeros([num_bin])

    for i,t in enumerate(threshold):
        predict = (output > t)
        P = target.sum().item()
        TP = predict[target == 1].sum().item()
        FN = P - TP
        N = target.numel() - P
        FP = predict.sum().item()- TP
        TN = N - FP
        if P > 0:
            sensitivity = TP / P
        else:
            sensitivity = 0
        if N > 0:
            specificity = TN / N
        else:
            specificity = 0
        sensitivity_list[i] = sensitivity
        specificity_list[i] = specificity
    AUC = np.trapz(sensitivity_list, x = specificity_list)
    t = 0.5
    predict = (output > t)
    P = target.sum().item()
    TP = predict[target == 1].sum().item()
    FN = P - TP
    N = target.numel() - P
    FP = predict.sum().item() - TP
    TN = N - FP
    if P > 0:
        sensitivity = TP / P
    else:
        sensitivity = 0
    if N > 0:
        specificity = TN / N
    else:
        specificity = 0
    return TP, TN, FP, FN, sensitivity, specificity,AUC

def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print('haha')

    data_train = pd.read_csv("data/trainingSet.csv")
    data_test = pd.read_csv("data/testingSet.csv")

    temp = np.mat(data_train)
    x_data = temp[:, 1:31].astype('float32')
    x_data_weight = temp[:,31].astype('float32')
    y_data = temp[:, -1] == 's'

    temp = np.mat(data_test)
    x_test = temp[:, 1:32].astype('float32')
    x_test_weight = temp[:, 31].astype('float32')
    y_test = temp[:, -1] == 's'



    permutation = np.random.choice(range(x_data.shape[0]), x_data.shape[0], replace=False)
    size_val = np.round(x_data.shape[0] * 0.2).astype(int)
    divid_index = np.arange(x_data.shape[0])

    num_bin = 100

    for cross_validate_index in range(5):
        print('cross validation:', cross_validate_index)
        lower_bound = cross_validate_index * size_val
        upper_bound = (cross_validate_index + 1) * size_val
        val_index = (divid_index >= lower_bound) * (divid_index < upper_bound)

        index_train = permutation[[not _ for _ in val_index]]
        index_val = permutation[val_index]

        x_train = x_data[index_train, :]
        y_train = y_data[index_train]
        x_train_weight = x_data_weight[index_train,:]

        x_val = x_data[index_val, :]
        y_val = y_data[index_val]
        x_val_weight = x_data_weight[index_val, :]

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.LongTensor(y_train).view(-1).to(device)
        x_train_weight = torch.FloatTensor(x_train_weight).view(-1).to(device)

        x_val = torch.FloatTensor(x_val).to(device)
        y_val = torch.LongTensor(y_val).view(-1).to(device)
        x_val_weight = torch.FloatTensor(x_val_weight).view(-1).to(device)

        ntrain = x_train.shape[0]
        nval = x_val.shape[0]
        dim = x_train.shape[1]

        num_hidden = args.n_hidden

        loss_func = nn.CrossEntropyLoss().to(device)

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.activation == 'tanh':
            net = Net_tanh(dim, num_hidden, 2)
        elif args.activation == 'relu':
            net = Net_relu(dim, num_hidden, 2)
        else:
            print('Unrecognized activation function')
            exit(0)
        net.to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        PATH = './result/' + 'cross_validate_' + str(cross_validate_index) + '/neural_network/'
        if not os.path.isdir(PATH):
            try:
                os.makedirs(PATH)
            except OSError as exc:  # Python >2.5
                if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                    pass
                else:
                    raise

        num_epochs = args.nepoch
        train_accuracy_path = np.zeros(num_epochs)
        train_loss_path = np.zeros(num_epochs)
        train_TP_path = np.zeros(num_epochs)
        train_auc_path = np.zeros(num_epochs)

        val_accuracy_path = np.zeros(num_epochs)
        val_loss_path = np.zeros(num_epochs)
        val_TP_path = np.zeros(num_epochs)
        val_auc_path = np.zeros(num_epochs)

        torch.manual_seed(args.seed)

        index = np.arange(ntrain)
        subn = args.batch_train

        for epoch in range(num_epochs):
            np.random.shuffle(index)
            for iter_index in range(ntrain // subn):
                subsample = index[(iter_index * subn):((iter_index + 1) * subn)]
                optimizer.zero_grad()
                output = net(x_train[subsample,])
                loss = (loss_func(output, y_train[subsample,]) * x_train_weight[subsample]).mean()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                print('epoch: ', epoch)
                output = net(x_train)
                probability = F.softmax(output, dim=1)[:, 1]
                TP, TN, FP, FN, sensitivity, specificity, AUC = cal_auc(probability, y_train, num_bin)
                train_loss = loss_func(output, y_train).mean()
                train_accuracy = (TP + TN) / ntrain
                train_loss_path[epoch] = train_loss
                train_accuracy_path[epoch] = train_accuracy
                train_TP_path[epoch] = TP
                train_auc_path[epoch] = AUC
                print("Training: Loss ", train_loss, 'Accuracy: ', train_accuracy, 'Sensitivity', sensitivity,
                      'Specificity', specificity, 'AUC', AUC)

                output = net(x_val)
                probability = F.softmax(output, dim=1)[:, 1]
                TP, TN, FP, FN, sensitivity, specificity, AUC = cal_auc(probability, y_val, num_bin)
                val_loss = loss_func(output, y_val).mean()
                val_accuracy = (TP + TN) / nval
                val_loss_path[epoch] = val_loss
                val_accuracy_path[epoch] = val_accuracy
                val_TP_path[epoch] = TP
                val_auc_path[epoch] = AUC
                print("Validation: Loss ", val_loss, 'Accuracy: ', val_accuracy, 'Sensitivity', sensitivity,
                      'Specificity', specificity, 'AUC', AUC)

            torch.save(net.state_dict(), PATH + 'model' + str(epoch) + '.pt')

        import pickle
        filename = PATH + 'result.txt'
        f = open(filename, 'wb')
        pickle.dump(
            [train_loss_path, train_accuracy_path, train_TP_path, train_auc_path, val_loss_path, val_accuracy_path, val_TP_path, val_auc_path], f)
        f.close()




if __name__ == '__main__':
    main()