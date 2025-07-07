import os
import sys
import json
import random
import os
import random
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from prettytable import PrettyTable
import torch


def metric_TP_FN(logit, truth):
    # prob = F.sigmoid(logit)
    _, prediction = torch.max(logit.data, dim=1)

    TP_FN = torch.sum(prediction == truth)
    return TP_FN

def train_one_epoch(model, optimizer, data_loader, device, batch_size):
    model.train()
    # loss_function = FocalLoss(gamma=0.5)
    loss_function = torch.nn.CrossEntropyLoss()

    train_loss_avg = torch.zeros(1).to(device)
    diag_TP_FN_sum = torch.zeros(1).to(device)
    sps_TP_FN_sum = torch.zeros(1).to(device)

    optimizer.zero_grad()
    total_num = len(data_loader.dataset)
    adj = torch.tensor(np.load('/home/ubuntu/zc/adj_npy.npy'))

    for step, (clinic_image, derm_image, label) in enumerate(data_loader):
        clinic_image = clinic_image.to(device)
        derm_image = derm_image.to(device)

        # Diagostic label
        diagnosis_label = label[0].long().to(device)
        # Seven-Point Checklikst labels
        pn_label = label[1].long().to(device)
        str_label = label[2].long().to(device)
        pig_label = label[3].long().to(device)
        rs_label = label[4].long().to(device)
        dag_label = label[5].long().to(device)
        bwv_label = label[6].long().to(device)
        vs_label = label[7].long().to(device)

        (diag, pn, str, pig, rs, dag, bwv, vs) = model(clinic_image.to(device), derm_image.to(device), adj.to(device))

        loss = torch.true_divide(
            loss_function(diag, diagnosis_label)
            + loss_function(pn, pn_label)
            + loss_function(str, str_label)
            + loss_function(pig, pig_label)
            + loss_function(rs, rs_label)
            + loss_function(dag, dag_label)
            + loss_function(bwv, bwv_label)
            + loss_function(vs, vs_label), 8)

        loss.backward()
        train_loss_avg = (train_loss_avg * step + loss.detach()) / (step + 1)  # update mean losses

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

        diag_TP_FN = metric_TP_FN(diag, diagnosis_label)
        sps_TP_FN = metric_TP_FN(pn, pn_label) + metric_TP_FN(str, str_label) + metric_TP_FN(pig, pig_label) \
                    + metric_TP_FN(rs, rs_label) + metric_TP_FN(dag, dag_label) + metric_TP_FN(bwv, bwv_label) \
                    + metric_TP_FN(vs, vs_label)
        diag_TP_FN_sum += diag_TP_FN
        sps_TP_FN_sum += sps_TP_FN
    train_acc_avg = (diag_TP_FN_sum + sps_TP_FN_sum) / (total_num * 8)
    train_acc_diag = diag_TP_FN_sum / total_num
    train_acc_sps = sps_TP_FN_sum / (total_num * 7)

    return train_loss_avg.item(), train_acc_avg.item(), train_acc_diag.item(), train_acc_sps.item()

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    # 验证样本总个数
    total_num = len(data_loader.dataset)
    # 用于存储预测正确的样本个数
    diag_TP_FN_sum = torch.zeros(1).to(device)
    sps_TP_FN_sum = torch.zeros(1).to(device)
    pn_TP_FN_sum = torch.zeros(1).to(device)
    str_TP_FN_sum = torch.zeros(1).to(device)
    pig_TP_FN_sum = torch.zeros(1).to(device)
    rs_TP_FN_sum = torch.zeros(1).to(device)
    dag_TP_FN_sum = torch.zeros(1).to(device)
    bwv_TP_FN_sum = torch.zeros(1).to(device)
    vs_TP_FN_sum = torch.zeros(1).to(device)

    adj = torch.tensor(np.load('/home/ubuntu/xiaochunlun2/Diffusion/adj_npy.npy'))

    for index, (clinic_image, derm_image, label) in enumerate(data_loader):
        clinic_image = clinic_image.to(device)
        derm_image = derm_image.to(device)

        diagnosis_label = label[0].long().to(device)
        pn_label = label[1].long().to(device)
        str_label = label[2].long().to(device)
        pig_label = label[3].long().to(device)
        rs_label = label[4].long().to(device)
        dag_label = label[5].long().to(device)
        bwv_label = label[6].long().to(device)
        vs_label = label[7].long().to(device)

        (diag, pn, str, pig, rs, dag, bwv, vs) = model(clinic_image.to(device), derm_image.to(device), adj.to(device))

        diag_TP_FN = metric_TP_FN(diag, diagnosis_label)
        pn_TP_FN = metric_TP_FN(pn, pn_label)
        str_TP_FN = metric_TP_FN(str, str_label)
        pig_TP_FN = metric_TP_FN(pig, pig_label)
        rs_TP_FN = metric_TP_FN(rs, rs_label)
        dag_TP_FN = metric_TP_FN(dag, dag_label)
        bwv_TP_FN = metric_TP_FN(bwv, bwv_label)
        vs_TP_FN = metric_TP_FN(vs, vs_label)

        diag_TP_FN_sum += diag_TP_FN
        pn_TP_FN_sum += pn_TP_FN
        str_TP_FN_sum += str_TP_FN
        pig_TP_FN_sum += pig_TP_FN
        rs_TP_FN_sum += rs_TP_FN
        dag_TP_FN_sum += dag_TP_FN
        bwv_TP_FN_sum += bwv_TP_FN
        vs_TP_FN_sum += vs_TP_FN
        sps_TP_FN_sum += pn_TP_FN + str_TP_FN + pig_TP_FN + rs_TP_FN + dag_TP_FN + bwv_TP_FN + vs_TP_FN

    acc_avg = round(100 * ((diag_TP_FN_sum + sps_TP_FN_sum) / (total_num * 8)).item(), 2)
    acc_diag = round(100 * (diag_TP_FN_sum / total_num).item(), 2)
    acc_sps = round(100 * (sps_TP_FN_sum / (total_num * 7)).item(), 2)
    acc_pn = round(100 * (pn_TP_FN_sum / total_num).item(), 2)
    acc_str = round(100 * (str_TP_FN_sum / total_num).item(), 2)
    acc_pig = round(100 * (pig_TP_FN_sum / total_num).item(), 2)
    acc_rs = round(100 * (rs_TP_FN_sum / total_num).item(), 2)
    acc_dag = round(100 * (dag_TP_FN_sum / total_num).item(), 2)
    acc_bwv = round(100 * (bwv_TP_FN_sum / total_num).item(), 2)
    acc_vs = round(100 * (vs_TP_FN_sum / total_num).item(), 2)

    task_names = ['Bwv', 'Dag', 'Pig', 'Pn', 'Rs', 'Str', 'Vs', 'Diag', 'SPS', 'Average']
    table = PrettyTable()
    table.field_names = ["", task_names[0], task_names[1], task_names[2], task_names[3], task_names[4], task_names[5],
                         task_names[6], task_names[7], task_names[8], task_names[9]]
    table.add_row(
        ['ACC', acc_bwv, acc_dag, acc_pig, acc_pn, acc_rs, acc_str, acc_vs, acc_diag, acc_sps, acc_avg])
    print(table)

    return acc_avg, acc_diag, acc_sps
