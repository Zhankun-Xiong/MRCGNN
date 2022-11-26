# -*- coding: utf-8 -*-
from datetime import datetime
import time 
import argparse
import copy
import torch
from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score,accuracy_score,recall_score,precision_score
import models
import custom_loss
from data_preprocessing import DrugDataset, DrugDataset1,DrugDataLoader, TOTAL_ATOM_FEATS
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

######################### Parameters ######################
parser = argparse.ArgumentParser()
parser.add_argument('--n_atom_feats', type=int, default=TOTAL_ATOM_FEATS, help='num of input features')
parser.add_argument('--n_atom_hid', type=int, default=64, help='num of hidden features')
parser.add_argument('--rel_total', type=int, default=65, help='num of interaction types')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=300, help='num of epochs')
parser.add_argument('--kge_dim', type=int, default=128, help='dimension of interaction matrix')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')

parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--neg_samples', type=int, default=1)
parser.add_argument('--data_size_ratio', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=True, choices=[0,1])
parser.add_argument('--zhongzi', type=int, default=0)


args = parser.parse_args()
n_atom_feats = args.n_atom_feats
n_atom_hid = args.n_atom_hid
rel_total = args.rel_total
lr = args.lr
n_epochs = args.n_epochs
kge_dim = args.kge_dim
batch_size = args.batch_size
zhongzi=args.zhongzi
weight_decay = args.weight_decay
neg_samples = args.neg_samples
data_size_ratio = args.data_size_ratio
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
print(args)

###### Dataset
df_ddi_train = pd.read_csv('data/'+str(zhongzi)+'/ddi_training1xiao.csv')
df_ddi_val = pd.read_csv('data/'+str(zhongzi)+'/ddi_validation1xiao.csv')
df_ddi_test = pd.read_csv('data/'+str(zhongzi)+'/ddi_test1xiao.csv')


train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]
val_tup = [(h, t, r) for h, t, r in zip(df_ddi_val['d1'], df_ddi_val['d2'], df_ddi_val['type'])]
test_tup = [(h, t, r) for h, t, r in zip(df_ddi_test['d1'], df_ddi_test['d2'], df_ddi_test['type'])]

train_data = DrugDataset(train_tup, ratio=data_size_ratio, neg_ent=neg_samples)
val_data = DrugDataset(val_tup, ratio=data_size_ratio, disjoint_split=False)
test_data = DrugDataset(test_tup, disjoint_split=False)
druglist=DrugDataset1(train_tup, ratio=data_size_ratio, neg_ent=neg_samples)

print(f"Training with {len(train_data)} samples, validating with {len(val_data)}, and testing with {len(test_data)}")

train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True,drop_last=True)
val_data_loader = DrugDataLoader(val_data, batch_size=batch_size ,drop_last=True)
test_data_loader = DrugDataLoader(test_data, batch_size=batch_size ,drop_last=False)
druglist=DrugDataLoader(druglist)

def compute_weigth(batch, device, training=True):
    '''
        *batch: (pos_tri, neg_tri)
        *pos/neg_tri: (batch_h, batch_t, batch_r)
    '''
    probas_pred, ground_truth = [], []
    for batch in druglist:
        pos_tri = batch
        pos_tri = [tensor.to(device=device) for tensor in pos_tri]

        h_weight = model.get_weight(pos_tri,zhongzi)
        break

    return h_weight

def do_compute(batch, device, model,training=True):

        probas_pred, ground_truth = [], []
        pos_tri, neg_tri ,hlist= batch


        pos_tri = [tensor.to(device=device) for tensor in pos_tri]

        p_score,gt = model(pos_tri)

        return p_score, gt


def do_compute_metrics(probas_pred, target):


    y_pred_train1 = []
    y_label_train = np.array(target)

    y_label_train=y_label_train.reshape((-1))
    y_pred_train = np.array(probas_pred).reshape((-1, 65))
    for i in range(y_pred_train.shape[0]):
        a = np.max(y_pred_train[i])
        #print(y_pred_train[i])
        #print(a)
        for j in range(y_pred_train.shape[1]):
            if y_pred_train[i][j] == a:
                # print(y_pred_train[i][j])
                y_pred_train1.append(j)
                break


    acc = accuracy_score(y_label_train, y_pred_train1)
    f1_score1 = f1_score(y_label_train, y_pred_train1, average='macro')
    recall1 = recall_score(y_label_train, y_pred_train1, average='macro')
    precision1 = precision_score(y_label_train, y_pred_train1, average='macro')
    aaa=y_label_train
    bbb=y_pred_train

    return acc, f1_score1, recall1,precision1,aaa,bbb


def train(model, train_data_loader, val_data_loader, loss_fn,  optimizer, n_epochs, device, scheduler=None):
    print('Starting training at', datetime.today())
    m = torch.nn.Sigmoid()
    maxacc=0
    for i in range(1, n_epochs+1):
        start = time.time()
        train_loss = 0
        train_loss_pos = 0
        train_loss_neg = 0
        val_loss = 0
        val_loss_pos = 0
        val_loss_neg = 0
        train_probas_pred = []
        train_ground_truth = []
        val_probas_pred = []
        val_ground_truth = []

        for batch in train_data_loader:
            #with torch.no_grad():
            #    drug_weight = compute_weigth(druglist, device)
            model.train()
            #print(batch)
            p_score, gt= do_compute(batch, device,model)

            loss = loss_fn(p_score, gt)
            # train_ground_truth1 = torch.tensor(train_ground_truth, dtype=torch.long, requires_grad=False)
            # train_probas_pred1 = torch.tensor(train_probas_pred, requires_grad=True)
            p_score=p_score.detach().cpu().numpy()
            gt=gt.detach().cpu().numpy()
            train_probas_pred.append(np.array(p_score))
            train_ground_truth.append(gt)

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            train_loss += loss.item() * len(p_score)


        train_loss /= len(train_data)

        with torch.no_grad():
            train_probas_pred = np.concatenate(train_probas_pred)
            train_ground_truth = np.concatenate(train_ground_truth)

            train_acc, train_f1, train_recall,train_precision,_,_ = do_compute_metrics(train_probas_pred, train_ground_truth)
            drug_weight=compute_weigth(druglist,device)

            for batch in val_data_loader:
                model.eval()
                probas_pred, gt,  = do_compute(batch, device,model,training=False)


                loss = loss_fn(probas_pred, gt)
                probas_pred = probas_pred.detach().cpu().numpy()
                gt = gt.detach().cpu().numpy()
                val_probas_pred.append(np.array(probas_pred))
                val_ground_truth.append(gt)

                val_loss += loss.item() * len(probas_pred)

            val_loss /= len(val_data)
            val_probas_pred = np.concatenate(val_probas_pred)
            val_ground_truth = np.concatenate(val_ground_truth)
            val_acc, val_f1, val_recall, val_precision ,_,_= do_compute_metrics(val_probas_pred, val_ground_truth)
            if scheduler:

                scheduler.step()
            if val_acc > maxacc:
                model_max = copy.deepcopy(model)
                maxacc = val_acc
            else:  #
                model_max = copy.deepcopy(model)





        print(f'Epoch: {i} ({time.time() - start:.4f}s), train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f},'
        f' train_acc: {train_acc:.4f}, val_acc:{val_acc:.4f}')
        print(f'\t\ttrain_f1: {train_f1:.4f}, val_f1: {val_f1:.4f}, train_val_recall: {train_recall:.4f}, val_recall: {val_recall:.4f}, train_precision: {train_precision:.4f}, val_precision: {val_precision:.4f}')

    test_probas_pred=[]
    test_ground_truth=[]
    test_loss=0
    with torch.no_grad():
        for batch in test_data_loader:
            model.eval()
            probas_pred, gt, = do_compute(batch, device,model_max)

            loss = loss_fn(probas_pred, gt)
            probas_pred = probas_pred.detach().cpu().numpy()
            gt = gt.detach().cpu().numpy()
            test_probas_pred.append(np.array(probas_pred))
            test_ground_truth.append(gt)

            test_loss += loss.item() * len(probas_pred)

        test_loss /= len(val_data)
        test_probas_pred = np.concatenate(test_probas_pred)
        test_ground_truth = np.concatenate(test_ground_truth)
        print(test_probas_pred.shape)
        print(test_ground_truth.shape)
        test_acc, test_f1, test_recall, test_precision ,y_label_train123,y_pred_train123= do_compute_metrics(test_probas_pred, test_ground_truth)

    if scheduler:
        # print('scheduling')
        scheduler.step()


    print(f'\t\t test_acc: {test_acc:.4f}, test_f1: {test_f1:.4f}, test_recall: {test_recall:.4f},test_precision: {test_precision:.4f}')
    with open('trimnet.txt', 'a') as f:


        f.write(str(zhongzi)+'  '+str(test_acc)+'  '+str(test_f1)+'  '+str(test_recall)+'  '+str(test_precision)+'\n')


model = models.TrimNet(55, 10, hidden_dim=64, depth=3,heads=4, dropout=0.2, outdim=1)

model.to(device=device)
loss=torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))



# if __name__ == '__main__':
train(model, train_data_loader, val_data_loader, loss, optimizer, n_epochs, device, scheduler)


