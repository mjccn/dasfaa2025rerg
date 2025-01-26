import os
import sys

sys.path.append(os.getcwd())
from process import *
from earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from rand5fold import *
from evaluate import *
import random
from model import *
import numpy as np
import time



def setup_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True


def validate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    interval = 5 * 60  # 5 minutes in seconds
    accuracies = []

    with th.no_grad():
        for Batch_data in test_loader:
            Batch_data.to(device)
            _, _, y_test = model(Batch_data)
            labels = th.cat((Batch_data.y1, Batch_data.y2), 0)
            _, predicted = th.max(y_test.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if time.time() - start_time >= interval:
                accuracy = 100 * correct / total
                accuracies.append(accuracy)
                start_time = time.time()

    accuracy = 100 * correct / total
    return accuracy, accuracies


def semi_grad_function(z1, z2, neg_matrix, ex_r):
    f = lambda x: th.exp(x / 0.5)
    sim_t = F.normalize(z1) @ F.normalize(z2).T
    sim1 = f(sim_t)
    sim2 = f(F.normalize(z1) @ F.normalize(z1).T)
    sim3 = f(F.normalize(z2) @ F.normalize(z2).T)
    l1 = -th.log(
        sim1.diag() / (((sim1 * neg_matrix).sum(dim=1) + (sim2 * neg_matrix).sum(dim=1)) * ex_r + sim1.diag()))
    l2 = -th.log(
        sim1.diag() / (((sim1 * neg_matrix).sum(dim=1) + (sim3 * neg_matrix).sum(dim=1)) * ex_r + sim1.diag()))

    return ((l1 + l2) / 2).mean() - sim_t.diag().mean() * (1 - 1 / ex_r)


def get_sim(model, x, edge_index, batch):
    with th.no_grad():
        model.eval()
        z = model.GCN_Net(x, edge_index, batch)
        return F.normalize(z) @ F.normalize(z).T


setup_seed(2022)


def train_rerg_twitter(x_test, x_train, lr, weight_decay, patience, n_epochs, batchsize, dataname):
    model = RERG(768, 64, 64).to(device)
    optimizer = th.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    model.train()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    beta = 0.001
    interval_accuracies = []

    for epoch in range(1, n_epochs + 1):
        traindata_list, testdata_list = loadData(dataname, x_train, x_test, droprate=0.4)
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=5, drop_last=False)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=5, drop_last=False)
        tqdm_train_loader = tqdm(train_loader)

        avg_loss = []
        avg_acc = []
        batch_idx = 0
        NUM = 1
        ex_r = 1

        for Batch_data in tqdm_train_loader:
            Batch_data.to(device)
            neg_matrix = th.ones(Batch_data.y1.size()[0], Batch_data.y1.size()[0])
            neg_matrix = neg_matrix.to(device)
            z1, z2, y_pre = model(Batch_data)
            y_lable = th.cat((Batch_data.y1, Batch_data.y2), 0)
            loss_cross = F.nll_loss(y_pre, y_lable)
            loss_cross = loss_cross.to(device)
            if epoch == 1:
                loss_contrastive = semi_grad_function(z1, z2, neg_matrix, ex_r)
                ori_sim = get_sim(model, Batch_data.x_orgin, Batch_data.edgeindex_origin, Batch_data.batch)
            else:
                loss_contrastive = semi_grad_function(z1, z2, neg_matrix, ex_r)

            if epoch % 20 == 0:
                with th.no_grad():
                    sim = get_sim(model, Batch_data.x_orgin, Batch_data.edgeindex_origin, Batch_data.batch)
                    if sim.shape != ori_sim.shape:
                        sim = F.adaptive_avg_pool2d(sim.unsqueeze(0), ori_sim.shape).squeeze(0)
                    sim_grad = sim - ori_sim
                    te = sim_grad.reshape(-1).sort()[0][int(sim_grad.size()[0] * sim_grad.size()[1] * 0.05)]
                    neg_matrix = th.zeros_like(sim_grad).to(device)
                    neg_matrix[sim_grad >= te] = 1
                    ex_r = sim.mean() / ((sim * neg_matrix).mean())
                    ori_sim = sim
            loss_contrastive = loss_contrastive.to(device)
            final_loss = loss_cross + beta * loss_contrastive
            avg_loss.append(final_loss.item())
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            _, pred = y_pre.max(dim=-1)
            correct = pred.eq(y_lable).sum().item()
            train_acc = correct / len(y_lable)
            avg_acc.append(train_acc)
            print("Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(epoch, batch_idx,
                                                                                                 final_loss.item(),
                                                                                                 train_acc))
            batch_idx = batch_idx + 1
            NUM += 1
        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        val_accuracy, interval_accs = validate(model, test_loader)
        interval_accuracies.extend(interval_accs)
        if val_accuracy < 50:
            beta *= 1.1
        else:
            beta *= 0.9
        beta = max(min(beta, 1), 0.01)

        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
            temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
            temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
            temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        model.eval()

        tqdm_test_loader = tqdm(test_loader)
        for Batch_data in tqdm_test_loader:
            Batch_data.to(device)
            _, _, y_test = model(Batch_data)
            lable_test = th.cat((Batch_data.y1, Batch_data.y2), 0)
            val_loss = F.nll_loss(y_test, lable_test)

            temp_val_losses.append(val_loss.item())
            _, val_pred = y_test.max(dim=1)
            correct = val_pred.eq(lable_test).sum().item()
            val_acc = correct / len(lable_test)

            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                val_pred, lable_test)

            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
                temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2), \
                temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                Recll3), temp_val_F3.append(F3), \
                temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
                Recll4), temp_val_F4.append(F4)
            temp_val_accs.append(val_acc)
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f} | Val_Accuracy {:.4f}".format(epoch, np.mean(avg_loss),
                                                                            np.mean(temp_val_losses),
                                                                            np.mean(temp_val_accs)))
        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
               'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                       np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        print('results:', res)

        if epoch > 25:
            early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
                           np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'RERG', dataname)
        accs = np.mean(temp_val_accs)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        F4 = np.mean(temp_val_F4)
        if early_stopping.early_stop:
            print("Early stopping")
            accs = early_stopping.accs
            F1 = early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            break

    print("Interval Accuracies: ", interval_accuracies)
    return accs, F1, F2, F3, F4


##---------------------------------main---------------------------------------
if __name__ == '__main__':
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    lr = 0.0005
    weight_decay = 1e-4
    patience = 10
    n_epochs = 200
    batchsize = 120

    datasetname = 'Twitter15'
    data_path = './data/twitter15/'
    label_path = './data/Twitter15_label_All.txt'

    test_accs = []
    NR_F1 = []  # NR
    FR_F1 = []  # FR
    TR_F1 = []  # TR
    UR_F1 = []  # UR

    fold0_x_test, fold0_x_train, \
        fold1_x_test, fold1_x_train, \
        fold2_x_test, fold2_x_train, \
        fold3_x_test, fold3_x_train, \
        fold4_x_test, fold4_x_train = load5foldData(datasetname, data_path, label_path)

    print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train))
    print('fold1 shape: ', len(fold1_x_test), len(fold1_x_train))
    print('fold2 shape: ', len(fold2_x_test), len(fold2_x_train))
    print('fold3 shape: ', len(fold3_x_test), len(fold3_x_train))
    print('fold4 shape: ', len(fold4_x_test), len(fold4_x_train))

    accs0, F1_0, F2_0, F3_0, F4_0 = train_rerg_twitter(fold0_x_test, fold0_x_train, lr, weight_decay, patience, n_epochs,
                                               batchsize, datasetname)
    accs1, F1_1, F2_1, F3_1, F4_1 = train_rerg_twitter(fold1_x_test, fold1_x_train, lr, weight_decay, patience, n_epochs,
                                               batchsize, datasetname)
    accs2, F1_2, F2_2, F3_2, F4_2 = train_rerg_twitter(fold2_x_test, fold2_x_train, lr, weight_decay, patience, n_epochs,
                                               batchsize, datasetname)
    accs3, F1_3, F2_3, F3_3, F4_3 = train_rerg_twitter(fold3_x_test, fold3_x_train, lr, weight_decay, patience, n_epochs,
                                               batchsize, datasetname)
    accs4, F1_4, F2_4, F3_4, F4_4 = train_rerg_twitter(fold4_x_test, fold4_x_train, lr, weight_decay, patience, n_epochs,
                                               batchsize, datasetname)
    test_accs.append((accs0 + accs1 + accs2 + accs3 + accs4) / 5)
    NR_F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)
    FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
    TR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
    UR_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)
    print(
        "AVG_result: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(sum(test_accs), sum(NR_F1),
                                                                                            sum(FR_F1), sum(TR_F1),
                                                                                            sum(UR_F1)))
