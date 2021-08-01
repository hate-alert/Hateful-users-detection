#!/usr/bin/env python
# coding: utf-8


import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import SGConv, ARMAConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import AGNNConv
import pickle
import json
import gzip
import numpy as np
import random
import time
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, confusion_matrix, recall_score, precision_score, f1_score, auc, accuracy_score



dataDirectory = '../Dataset/GabData/'

with open(dataDirectory+'GAB_Data/haters.json') as json_file:
    haters = json.load(json_file)

with open(dataDirectory+'GAB_Data/nonhaters.json') as json_file:
    non_haters = json.load(json_file)

print("User Info loading Done")



with open(dataDirectory +'twitterTrain_GABDoc2vec_100.p', 'rb') as handle:
    doc_vectors=pickle.load(handle)
print("Doc Vector Loading Done")



with gzip.open(dataDirectory + 'gabEdges1_5degree.pklgz') as fp:
    final_list= pickle.load(fp)

print("NetWork Loading Done")



graph={}
graph_dict={}
inv_graph_dict = {}
nodes=0

for i in final_list:
    if i[0] not in graph:
        graph[i[0]]=[]
        graph_dict[i[0]]=nodes
        inv_graph_dict[nodes] = i[0];
        nodes+=1
    if i[1] not in graph:
        graph[i[1]]=[]
        graph_dict[i[1]]=nodes
        inv_graph_dict[nodes] = i[1];
        nodes+=1
    graph[i[0]].append(i[1])

print("Number of Users in the Network:", nodes)



rows=[]
cols=[]

for elem in graph_dict:
    neighbours= graph[elem]
    r=graph_dict[elem]
    for neighbour in neighbours:
        c = graph_dict[neighbour]
        rows.append(r)
        cols.append(c)

edge_index= torch.LongTensor([rows,cols])

print("Edge Index Created")



_X= []
_y =[]

for i in range(0,nodes):
    _X.append(doc_vectors[inv_graph_dict[i]])
    if inv_graph_dict[i] in haters:
        _y.append(1)
    elif inv_graph_dict[i] in non_haters:
        _y.append(0)
    else:
        _y.append(2)

featureVector = torch.FloatTensor(_X)
labels        = torch.LongTensor(_y)



num_features = 100
num_classes = 2

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = torch.nn.Linear(num_features, 32)
        self.prop1 = AGNNConv(requires_grad=True)
        self.lin2 = torch.nn.Linear(32, 2)

    def forward(self):
        x=X
        x = F.dropout(x, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.prop1(x, edge_index)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

    
def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[train_mask], y[train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    logits    = model()
    accs      = []
    Mf1_score = []
    for mask in [train_mask, test_mask]:
        pred = logits[mask].max(1)[1]
        acc  = pred.eq(y[mask]).sum().item() / mask.sum().item()
        mfc  = f1_score(y[mask].detach().cpu(), pred.detach().cpu(), average='macro')
        accs.append(acc)
        Mf1_score.append(mfc)
    return accs, Mf1_score



def getData():
    return featureVector, labels

def ratio_split(haters,non_haters,nodes):
    testList = list(haters)
    testList.extend(non_haters)
    test_mask  = [0] * nodes
    
    for user in testList:
        test_mask[graph_dict[user]]=1
    test_mask  = torch.ByteTensor(test_mask)
    return test_mask



#device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
X, y = getData()
test_mask = ratio_split(haters,non_haters, nodes)
edge_index= edge_index.to(device)
y= y.to(device)
X= X.to(device)
test_mask= test_mask.to(device)



def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = Net()
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model

def evalMetric(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    mf1Score = f1_score(y_true, y_pred, average='macro')
    f1Score  = f1_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    area_under_c = auc(fpr, tpr)
    recallScore = recall_score(y_true, y_pred)
    precisionScore = precision_score(y_true, y_pred)
    return {"accuracy": accuracy, 'mF1Score': mf1Score, 'f1Score': f1Score, 'auc': area_under_c, 
            'precision': precisionScore, 'recall': recallScore}




model = load_checkpoint(dataDirectory + 'TwitterAGNNCheckpoint.pth')
logits, accs = model(), []
pred = logits[test_mask].max(1)[1]

# Evaluation Metric
print("Eval Metric")
print(evalMetric(y[test_mask], pred))
#Classification Report
print("\nClassification Report")
print(classification_report(y[test_mask], pred))

#Confusion Matrix
print("\nConfusion Matrix")
print(confusion_matrix(y[test_mask], pred))




