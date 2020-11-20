import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import errno
import pandas as pd
import matplotlib.pyplot as plt
import argparse



parser = argparse.ArgumentParser(description='Higgs Boson')
parser.add_argument('--nepoch', default = 300, type = int, help = 'total number of training epochs')
parser.add_argument('--lr', default = 0.001, type = float, help = 'initial learning rate')
parser.add_argument('--momentum', default = 0.9, type = float, help = 'momentum in SGD')
parser.add_argument('--weight_decay', default = 0.0001, type = float, help = 'weight decay in SGD')
parser.add_argument('--batch_train', default = 128, type = int, help = 'batch size for training')

parser.add_argument('--n_hidden_list', default=[1,2,3], type = int, nargs='+', help = 'number of hidden layer')
parser.add_argument('--activation_list', default=['tanh','relu'], type = str, nargs='+', help = 'set activation function')


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


def cal_auc(output, target, num_bin, weight, final_threshold = 0.5, plot = False, plot_label = None):
    min_val = output.min().item()
    max_val = output.max().item()
    threshold = np.linspace(min_val, max_val, num_bin)
    sensitivity_list = np.zeros([num_bin])
    specificity_list = np.zeros([num_bin])

    for i,t in enumerate(threshold):
        predict = (output > t)
        label = (target == 1)
        # P = target.sum().item()
        # TP = predict[target == 1].sum().item()
        # FN = P - TP
        # N = target.numel() - P
        # FP = predict.sum().item()- TP
        # TN = N - FP

        P = weight[label].sum().item()
        TP = weight[label * predict].sum().item()
        FN = P - TP
        N = weight.sum().item() - P
        FP = weight[predict].sum().item() - TP
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

    if plot:
        plt.plot(1 - specificity_list, sensitivity_list, label = plot_label)

    t = final_threshold

    predict = (output > t)
    label = (target == 1)

    P = weight[label].sum().item()
    TP = weight[label * predict].sum().item()
    FN = P - TP
    N = weight.sum().item() - P
    FP = weight[predict].sum().item() - TP
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



def cal_AMS(output, target, num_bin, weight):
    min_val = output.min().item()
    max_val = output.max().item()
    threshold = np.linspace(min_val, max_val, num_bin)
    sensitivity_list = np.zeros([num_bin])
    specificity_list = np.zeros([num_bin])

    unnorm_TPR = np.zeros([num_bin])
    unnorm_FPR = np.zeros([num_bin])

    for i,t in enumerate(threshold):
        predict = (output > t)
        label = (target == 1)
        # P = target.sum().item()
        # TP = predict[target == 1].sum().item()
        # FN = P - TP
        # N = target.numel() - P
        # FP = predict.sum().item()- TP
        # TN = N - FP

        P = weight[label].sum().item()
        TP = weight[label * predict].sum().item()
        FN = P - TP
        N = weight.sum().item() - P
        FP = weight[predict].sum().item() - TP
        TN = N - FP

        unnorm_TPR[i] = TP
        unnorm_FPR[i] = FP

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
    br = 10
    AMS = np.sqrt(2 * ((unnorm_TPR + unnorm_FPR + br)*np.log(1 + unnorm_TPR / (unnorm_FPR + br)) - unnorm_TPR))
    index = np.argmax(AMS)
    t = threshold[index]
    
    return AUC, t




def kfold_cv(x_data, y_data, x_data_weight, num_hidden, activation,  lr = 0.001, momentum = 0.9, weight_decay =0.0001, nepoch = 300, subn = 128, num_bin = 100, init_seed = 1):

    train_auc_list = np.zeros(5)
    val_auc_list = np.zeros(5)
    best_epoch_list = np.zeros(5)

    permutation = np.random.choice(range(x_data.shape[0]), x_data.shape[0], replace=False)
    size_val = np.round(x_data.shape[0] * 0.2).astype(int)
    divid_index = np.arange(x_data.shape[0])

    plt.figure(1)
    plt.clf()
    plt.figure(2)
    plt.clf()

    for cross_validate_index in range(5):
        seed = init_seed + cross_validate_index
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

        loss_func = nn.CrossEntropyLoss(reduction='none').to(device)

        np.random.seed(seed)
        torch.manual_seed(seed)
        if activation == 'tanh':
            net = Net_tanh(dim, num_hidden, 2)
        elif activation == 'relu':
            net = Net_relu(dim, num_hidden, 2)
        else:
            print('Unrecognized activation function')
            exit(0)
        net.to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                                    weight_decay=weight_decay)

        PATH = './result/neural_network/' + activation + '/nlayer' + str(num_hidden) + '/' + 'cross_validate_' + str(
            cross_validate_index) + '/'
        if not os.path.isdir(PATH):
            try:
                os.makedirs(PATH)
            except OSError as exc:  # Python >2.5
                if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                    pass
                else:
                    raise

        num_epochs = nepoch
        train_accuracy_path = np.zeros(num_epochs)
        train_loss_path = np.zeros(num_epochs)
        train_TP_path = np.zeros(num_epochs)
        train_auc_path = np.zeros(num_epochs)

        val_accuracy_path = np.zeros(num_epochs)
        val_loss_path = np.zeros(num_epochs)
        val_TP_path = np.zeros(num_epochs)
        val_auc_path = np.zeros(num_epochs)

        torch.manual_seed(seed)

        index = np.arange(ntrain)

        best_val_auc = 0
        best_epoch = 0
        stop_flag = False

        for epoch in range(num_epochs):
            if stop_flag:
                break
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
                TP, TN, FP, FN, sensitivity, specificity, AUC = cal_auc(probability, y_train, num_bin, x_train_weight)
                train_loss = loss_func(output, y_train).mean()
                train_accuracy = (TP + TN) / x_train_weight.sum()
                train_loss_path[epoch] = train_loss
                train_accuracy_path[epoch] = train_accuracy
                train_TP_path[epoch] = TP
                train_auc_path[epoch] = AUC
                print("Training: Loss ", train_loss, 'Accuracy: ', train_accuracy, 'Sensitivity', sensitivity,
                      'Specificity', specificity, 'AUC', AUC)

                output = net(x_val)
                probability = F.softmax(output, dim=1)[:, 1]
                TP, TN, FP, FN, sensitivity, specificity, AUC = cal_auc(probability, y_val, num_bin, x_val_weight)
                val_loss = loss_func(output, y_val).mean()
                val_accuracy = (TP + TN) / x_val_weight.sum()
                val_loss_path[epoch] = val_loss
                val_accuracy_path[epoch] = val_accuracy
                val_TP_path[epoch] = TP
                val_auc_path[epoch] = AUC
                print("Validation: Loss ", val_loss, 'Accuracy: ', val_accuracy, 'Sensitivity', sensitivity,
                      'Specificity', specificity, 'AUC', AUC)
                if AUC > best_val_auc:
                    best_val_auc = AUC
                    best_epoch = epoch
                else:
                    if best_epoch < epoch - 10:
                        stop_flag = True
                        break

            torch.save(net.state_dict(), PATH + 'model' + str(epoch) + '.pt')
        train_auc_list[cross_validate_index] = train_auc_path[best_epoch]
        val_auc_list[cross_validate_index] = val_auc_path[best_epoch]
        best_epoch_list[cross_validate_index] = best_epoch

        import pickle
        filename = PATH + 'result.txt'
        f = open(filename, 'wb')
        pickle.dump(
            [train_loss_path, train_accuracy_path, train_TP_path, train_auc_path, val_loss_path,
             val_accuracy_path, val_TP_path, val_auc_path], f)
        f.close()
        
        
        net.load_state_dict(torch.load(PATH + 'model' + str(best_epoch) + '.pt'))
        with torch.no_grad():
            print('best epoch: ', best_epoch)
            
            plt.figure(1)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')

            output = net(x_train)
            probability = F.softmax(output, dim=1)[:, 1]
            AUC, final_threshold = cal_AMS(probability, y_train, num_bin, x_train_weight)
            train_TP, train_TN, train_FP, train_FN, train_sensitivity, train_specificity, train_AUC = cal_auc(
                probability, y_train, num_bin, x_train_weight, final_threshold, plot=True, plot_label=('Training: CV' + str(cross_validate_index)))
            train_loss = loss_func(output, y_train).mean()
            train_accuracy = (train_TP + train_TN) / x_train_weight.sum()
            print("Training: Loss ", train_loss, 'Accuracy: ', train_accuracy, 'Sensitivity', train_sensitivity,
                  'Specificity', train_specificity, 'AUC', train_AUC)


            plt.figure(2)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')

            output = net(x_val)
            probability = F.softmax(output, dim=1)[:, 1]
            val_TP, val_TN, val_FP, val_FN, val_sensitivity, val_specificity, val_AUC = cal_auc(probability,
                                                                                                       y_val, num_bin,
                                                                                                       x_val_weight,
                                                                                                       final_threshold,
                                                                                                       plot=True,
                                                                                                       plot_label=('Validation: CV' + str(cross_validate_index)))
            val_loss = loss_func(output, y_val).mean()
            val_accuracy = (val_TP + val_TN) / x_val_weight.sum()
            print("val: Loss ", val_loss, 'Accuracy: ', val_accuracy, 'Sensitivity', val_sensitivity,
                  'Specificity', val_specificity, 'AUC', val_AUC)
        PATH = './result/neural_network/' + activation + '/nlayer' + str(num_hidden) + '/'
        plt.figure(1)
        plt.legend(loc='lower right')
        plt.savefig(PATH + 'train_roc_curve.png')
        plt.figure(2)
        plt.legend(loc='lower right')
        plt.savefig(PATH + 'valid_roc_curve.png')
        
    return train_auc_list, val_auc_list, best_epoch_list



def test(x_data, y_data, x_data_weight, x_test, y_test, x_test_weight, num_hidden, activation,  lr = 0.001, momentum = 0.9, weight_decay =0.0001, nepoch = 300, subn = 128, num_bin = 100, seed = 1):

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    ndata = x_data.shape[0]
    index = np.arange(ndata)
    np.random.shuffle(index)

    train_size = int(round(ndata * 0.9))

    x_train = x_data[index[0:train_size]]
    y_train = y_data[index[0:train_size]]
    x_train_weight = x_data_weight[index[0:train_size]]

    x_val = x_data[index[train_size:]]
    y_val = y_data[index[train_size:]]
    x_val_weight = x_data_weight[index[train_size:]]


    x_train = torch.FloatTensor(x_train).to(device)
    y_train = torch.LongTensor(y_train).view(-1).to(device)
    x_train_weight = torch.FloatTensor(x_train_weight).view(-1).to(device)
    
    x_val = torch.FloatTensor(x_val).to(device)
    y_val = torch.LongTensor(y_val).view(-1).to(device)
    x_val_weight = torch.FloatTensor(x_val_weight).view(-1).to(device)


    x_test = torch.FloatTensor(x_test).to(device)
    y_test = torch.LongTensor(y_test).view(-1).to(device)
    x_test_weight = torch.FloatTensor(x_test_weight).view(-1).to(device)

    ntrain = x_train.shape[0]
    dim = x_train.shape[1]

    loss_func = nn.CrossEntropyLoss(reduction='none').to(device)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if activation == 'tanh':
        net = Net_tanh(dim, num_hidden, 2)
    elif activation == 'relu':
        net = Net_relu(dim, num_hidden, 2)
    else:
        print('Unrecognized activation function')
        exit(0)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay)

    PATH = './result/neural_network/test/train_size' + str(ntrain) + '/'
    if not os.path.isdir(PATH):
        try:
            os.makedirs(PATH)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                pass
            else:
                raise

    num_epochs = nepoch
    train_accuracy_path = np.zeros(num_epochs)
    train_loss_path = np.zeros(num_epochs)
    train_TP_path = np.zeros(num_epochs)
    train_auc_path = np.zeros(num_epochs)
    
    val_accuracy_path = np.zeros(num_epochs)
    val_loss_path = np.zeros(num_epochs)
    val_TP_path = np.zeros(num_epochs)
    val_auc_path = np.zeros(num_epochs)

    test_accuracy_path = np.zeros(num_epochs)
    test_loss_path = np.zeros(num_epochs)
    test_TP_path = np.zeros(num_epochs)
    test_auc_path = np.zeros(num_epochs)

    torch.manual_seed(seed)

    index = np.arange(ntrain)

    best_val_auc = 0
    best_epoch = 0
    stop_flag = False

    for epoch in range(num_epochs):
        if stop_flag:
            break
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
            TP, TN, FP, FN, sensitivity, specificity, AUC = cal_auc(probability, y_train, num_bin, x_train_weight)
            train_loss = loss_func(output, y_train).mean()
            train_accuracy = (TP + TN) / x_train_weight.sum()
            train_loss_path[epoch] = train_loss
            train_accuracy_path[epoch] = train_accuracy
            train_TP_path[epoch] = TP
            train_auc_path[epoch] = AUC
            print("Training: Loss ", train_loss, 'Accuracy: ', train_accuracy, 'Sensitivity', sensitivity,
                  'Specificity', specificity, 'AUC', AUC)
            


            output = net(x_test)
            probability = F.softmax(output, dim=1)[:, 1]
            TP, TN, FP, FN, sensitivity, specificity, AUC = cal_auc(probability, y_test, num_bin, x_test_weight)
            test_loss = loss_func(output, y_test).mean()
            test_accuracy = (TP + TN) / x_test_weight.sum()
            test_loss_path[epoch] = test_loss
            test_accuracy_path[epoch] = test_accuracy
            test_TP_path[epoch] = TP
            test_auc_path[epoch] = AUC
            print("Test: Loss ", test_loss, 'Accuracy: ', test_accuracy, 'Sensitivity', sensitivity,
                  'Specificity', specificity, 'AUC', AUC)

            output = net(x_val)
            probability = F.softmax(output, dim=1)[:, 1]
            TP, TN, FP, FN, sensitivity, specificity, AUC = cal_auc(probability, y_val, num_bin, x_val_weight)
            val_loss = loss_func(output, y_val).mean()
            val_accuracy = (TP + TN) / x_val_weight.sum()
            val_loss_path[epoch] = val_loss
            val_accuracy_path[epoch] = val_accuracy
            val_TP_path[epoch] = TP
            val_auc_path[epoch] = AUC
            print("Validation: Loss ", val_loss, 'Accuracy: ', val_accuracy, 'Sensitivity', sensitivity,
                  'Specificity', specificity, 'AUC', AUC)
            if AUC > best_val_auc:
                best_val_auc = AUC
                best_epoch = epoch
            else:
                if best_epoch < epoch - 10:
                    stop_flag = True
                    break

        torch.save(net.state_dict(), PATH + 'model' + str(epoch) + '.pt')
  

    import pickle
    filename = PATH + 'result.txt'
    f = open(filename, 'wb')
    pickle.dump(
        [train_loss_path, train_accuracy_path, train_TP_path, train_auc_path, val_loss_path, val_accuracy_path, val_TP_path, val_auc_path, test_loss_path,
         test_accuracy_path, test_TP_path, test_auc_path], f)
    f.close()

    net.load_state_dict(torch.load(PATH + 'model' + str(best_epoch) + '.pt'))

    x_train = x_data
    y_train = y_data
    x_train_weight = x_data_weight

    x_train = torch.FloatTensor(x_train).to(device)
    y_train = torch.LongTensor(y_train).view(-1).to(device)
    x_train_weight = torch.FloatTensor(x_train_weight).view(-1).to(device)

    with torch.no_grad():
        print('best epoch: ', best_epoch)

        plt.clf()
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        output = net(x_train)
        probability = F.softmax(output, dim=1)[:, 1]
        AUC, final_threshold = cal_AMS(probability, y_train, num_bin, x_train_weight)
        train_TP, train_TN, train_FP, train_FN, train_sensitivity, train_specificity, train_AUC = cal_auc(probability, y_train, num_bin, x_train_weight, final_threshold, plot=True, plot_label='Training')
        train_loss = loss_func(output, y_train).mean()
        train_accuracy = (train_TP + train_TN) / x_train_weight.sum()
        print("Training: Loss ", train_loss, 'Accuracy: ', train_accuracy, 'Sensitivity', train_sensitivity,
              'Specificity', train_specificity, 'AUC', train_AUC)


        output = net(x_test)
        probability = F.softmax(output, dim=1)[:, 1]
        test_TP, test_TN, test_FP, test_FN, test_sensitivity, test_specificity, test_AUC = cal_auc(probability, y_test, num_bin, x_test_weight, final_threshold, plot= True, plot_label='Testing')
        test_loss = loss_func(output, y_test).mean()
        test_accuracy = (test_TP + test_TN) / x_test_weight.sum()
        print("Test: Loss ", test_loss, 'Accuracy: ', test_accuracy, 'Sensitivity', test_sensitivity,
              'Specificity', test_specificity, 'AUC', test_AUC)

    plt.legend(loc='lower right')
    plt.savefig(PATH + 'roc_curve.png')

    return train_TP, train_TN, train_FP, train_FN, train_accuracy, train_AUC, test_TP, test_TN, test_FP, test_FN, test_accuracy, test_AUC


def acc_vs_training_sample(x_data, y_data, x_data_weight, x_test, y_test, x_test_weight, num_hidden, activation,  lr = 0.001, momentum = 0.9, weight_decay =0.0001, nepoch = 300, subn = 128, num_bin = 100, seed = 1):
    ntrain = x_data.shape[0]
    index = np.arange(ntrain)
    np.random.shuffle(index)
    train_size_list =  np.arange(1000, ntrain + 1, 1000)

    train_acc_list = np.zeros(len(train_size_list))
    test_acc_list = np.zeros(len(train_size_list))
    for train_index, train_size in enumerate(train_size_list):
        print('Training Size:', train_size)
        x_train = x_data[index[0:train_size]]
        y_train = y_data[index[0:train_size]]
        x_train_weight = x_data_weight[index[0:train_size]]
        train_TP, train_TN, train_FP, train_FN, train_accuracy, train_AUC, test_TP, test_TN, test_FP, test_FN, test_accuracy, test_AUC = test(x_train, y_train, x_train_weight, x_test, y_test, x_test_weight, num_hidden, activation,  lr = 0.001, momentum = 0.9, weight_decay =0.0001, nepoch = 300, subn = 128, num_bin = 100, seed = 1)
        train_acc_list[train_index] = train_accuracy
        test_acc_list[train_index] = test_accuracy


    PATH = './result/neural_network/test/'
    plt.clf()
    # plt.plot(train_size_list, train_acc_list, label = 'Training')
    plt.plot(train_size_list, test_acc_list, label = 'Testing')
    plt.xlabel('Number of Training Sample')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(PATH + 'acc_vs_size.png')

    return train_acc_list, test_acc_list




def main():
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_train = pd.read_csv("data/trainingSet.csv")
    data_test = pd.read_csv("data/testingSet.csv")

    temp = np.mat(data_train)
    x_data = temp[:, 1:31].astype('float32')
    x_data_weight = temp[:,31].astype('float32')
    y_data = temp[:, -1] == 's'

    temp = np.mat(data_test)
    x_test = temp[:, 1:31].astype('float32')
    x_test_weight = temp[:, 31].astype('float32')
    y_test = temp[:, -1] == 's'

    # num_hidden_list = [1, 2, 3]
    # activation_list = ['tanh', 'relu']
    # lr = 0.001
    # momentum = 0.9
    # weight_decay = 0.0001
    # nepoch = 300
    # subn = 128
    # num_bin = 100

    num_hidden_list = args.n_hidden_list
    activation_list = args.activation_list
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    nepoch = args.nepoch
    subn = args.batch_train
    num_bin = 100

    train_auc_result = [[] for _ in range(len(num_hidden_list))]
    val_auc_result = [[] for _ in range(len(num_hidden_list))]
    best_epoch_result = [[] for _ in range(len(num_hidden_list))]

    train_auc_mean = np.zeros([len(num_hidden_list), len(activation_list)])
    train_auc_std = np.zeros([len(num_hidden_list), len(activation_list)])
    val_auc_mean = np.zeros([len(num_hidden_list), len(activation_list)])
    val_auc_std = np.zeros([len(num_hidden_list), len(activation_list)])

    for num_hidden_index, num_hidden in enumerate(num_hidden_list):
        for activation_index, activation in enumerate(activation_list):
            train_auc_list, val_auc_list, best_epoch_list = kfold_cv(x_data, y_data, x_data_weight, num_hidden, activation, lr=lr, momentum=momentum, weight_decay=weight_decay,
                     nepoch=nepoch, subn=subn, num_bin = num_bin, init_seed = num_hidden_index * len(activation_list) + activation_index + 1)
            train_auc_result[num_hidden_index].append(train_auc_list)
            val_auc_result[num_hidden_index].append(val_auc_list)
            best_epoch_result[num_hidden_index].append(best_epoch_list)

            train_auc_mean[num_hidden_index, activation_index] = train_auc_list.mean()
            train_auc_std[num_hidden_index, activation_index] = train_auc_list.std()
            val_auc_mean[num_hidden_index, activation_index] = val_auc_list.mean()
            val_auc_std[num_hidden_index, activation_index] = val_auc_list.std()
    PATH = './result/neural_network/'
    import pickle
    filename = PATH + 'result.txt'
    f = open(filename, 'wb')
    pickle.dump(
        [train_auc_result, val_auc_result, best_epoch_result, train_auc_mean, train_auc_std,
         val_auc_mean, val_auc_std], f)
    f.close()

    temp = val_auc_mean.argmin()
    best_num_hidden_index = temp // len(activation_list)
    best_activation_index = temp % len(activation_list)

    best_num_hidden = num_hidden_list[best_num_hidden_index]
    best_activation = activation_list[best_activation_index]

    train_TP, train_TN, train_FP, train_FN, train_accuracy, train_AUC, test_TP, test_TN, test_FP, test_FN, test_accuracy, test_AUC = test(x_data, y_data, x_data_weight, x_test, y_test, x_test_weight, best_num_hidden, best_activation,  lr = 0.001, momentum = 0.9, weight_decay =0.0001, nepoch = 300, subn = 128, num_bin = 100)

    PATH = './result/neural_network/'
    import pickle
    filename = PATH + 'test_result.txt'
    f = open(filename, 'wb')
    pickle.dump(
        [train_TP, train_TN, train_FP, train_FN, train_accuracy, train_AUC, test_TP, test_TN, test_FP, test_FN, test_accuracy, test_AUC], f)
    f.close()

    train_acc_list, test_acc_list = acc_vs_training_sample(x_data, y_data, x_data_weight, x_test, y_test, x_test_weight, num_hidden, activation,  lr = 0.001, momentum = 0.9, weight_decay =0.0001, nepoch = 300, subn = 128, num_bin = 100, seed = 1)

    PATH = './result/neural_network/'
    import pickle
    filename = PATH + 'different_size.txt'
    f = open(filename, 'wb')
    pickle.dump(
        [train_acc_list, test_acc_list], f)
    f.close()



if __name__ == '__main__':
    main()
