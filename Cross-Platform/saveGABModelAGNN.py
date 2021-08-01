#!/usr/bin/env python
# coding: utf-8


import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import SGConv, ARMAConv, SAGEConv, AGNNConv
import pickle
import numpy as np
import random
import time
import sys
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, confusion_matrix, recall_score, precision_score, f1_score, auc, accuracy_score
import json
import gzip



dataDirectory = '../Dataset/GabData/'

with open(dataDirectory+'GAB_Data/haters.json') as json_file:
    haters = json.load(json_file)

with open(dataDirectory+'GAB_Data/nonhaters.json') as json_file:
    non_haters = json.load(json_file)
print("User Info loading Done")



with open(dataDirectory+'GABDoc2vec100.p', 'rb') as handle:
    doc_vectors=pickle.load(handle)

print("Doc_vec loading done")



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



def getData():
    return featureVector, labels



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



edge_index= torch.LongTensor([rows,cols])
print("Edges: ", len(rows))



def Diff(li1, li2): 
    return (list(set(li1) - set(li2))) 

def ratio_split(train_haters, train_non_haters, test_haters, test_non_haters, nodes):

    #Creating Training List
    trainList = list(train_haters)
    trainList.extend(train_non_haters)

    #Creating Testing DataPoint
    textList = list(test_haters)
    textList.extend(test_non_haters)
    
    train_mask = [0] * nodes
    test_mask  = [0] * nodes

    
    for i in trainList:
          train_mask[graph_dict[i]] = 1;
    
    for i in textList:
          test_mask[graph_dict[i]] = 1;

    train_mask = torch.ByteTensor(train_mask)
    test_mask  = torch.ByteTensor(test_mask)
    print("Splitting done")
    return train_mask, test_mask



with open(dataDirectory+'GAB_Data/hateval1.json') as json_file:
    test_haters = json.load(json_file)
with open(dataDirectory+'GAB_Data/nonhateval1.json') as json_file:
    test_non_haters = json.load(json_file)

train_haters     = Diff(haters, test_haters) 
train_non_haters = Diff(non_haters, test_non_haters)


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



random.seed(101)
X, y = getData()
train_mask, test_mask = ratio_split(train_haters, train_non_haters, test_haters, test_non_haters, nodes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

model  = Net().to(device)
edge_index= edge_index.to(device)
y= y.to(device)
X= X.to(device)
train_mask= train_mask.to(device)
test_mask= test_mask.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
best_val_acc = best_MfScore = train_acc = train_mfscore = 0
for epoch in range(1, 50):
    train()
    Accuracy, F1Score= test()
    if Accuracy[1] > best_val_acc:
        best_val_acc = Accuracy[1]
        train_acc     = Accuracy[0]
        best_MfScore = F1Score[1]
        train_mfscore = F1Score[0]
        #checkpoint = {'model': Net(), 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}
        #torch.save(checkpoint, 'GABcheckpoint_Final_.pth', pickle_protocol =3)
        checkpoint = {'state_dict': model.state_dict(),'optimizer' :optimizer.state_dict()}
        torch.save(checkpoint, 'GABAGNNCheckpoint.pth')



test()

print(best_val_acc)
print(best_MfScore)
print('Trainig')
print(train_acc)
print(train_mfscore)
