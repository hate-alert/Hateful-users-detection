#!/usr/bin/env python


#To install all the relavants package

get_ipython().system('pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html')
get_ipython().system('pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html')
get_ipython().system('pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html')
get_ipython().system('pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html')
get_ipython().system('pip install torch-geometric')
get_ipython().system('pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html')



import os.path as osp
from torch_geometric.nn import GCNConv, ChebConv
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
from torch_geometric.nn import SGConv, ARMAConv, SAGEConv, AGNNConv
from torch_geometric.nn import ChebConv
import pickle
import json
import gzip
import numpy as np
import random
import time
import sys
from sklearn.metrics import *



# Read the hateful users and non hateful users

dataDirectory = '../Dataset/GabData/'
with open(dataDirectory+'GAB_Data/haters.json') as json_file:
    haters = json.load(json_file)

with open(dataDirectory+'GAB_Data/nonhaters.json') as json_file:
    non_haters = json.load(json_file)
print("User Info loading Done")


# Read the Doc2vec embedding of all the users 
with open(dataDirectory+'GABDoc2vec100.p', 'rb') as handle:
    doc_vectors=pickle.load(handle)
print("Doc Vector Loading Done")

# To 
with gzip.open(dataDirectory+'gabEdges1_5degree.pklgz') as fp:
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



X= []
y =[]

for i in range(0,nodes):
  X.append(doc_vectors[inv_graph_dict[i]])
  if inv_graph_dict[i] in haters:
    y.append(1)
  elif inv_graph_dict[i] in non_haters:
    y.append(0)
  else:
    y.append(2)

featureVector     = torch.FloatTensor(X)
labels     = torch.LongTensor(y)



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




validationFold  = ['1', '2', '3', '4' ,'5']
percentages     = [0.05, 0.1, 0.15, 0.20, 0.5, 0.8]


def Diff(li1, li2): 
    return (list(set(li1) - set(li2))) 

def ratio_split(percentage, train_haters, train_non_haters, test_haters, test_non_haters, nodes):
    np.random.shuffle(train_haters)
    np.random.shuffle(train_non_haters)

    #Calculate Total number of training point and split training Data Point
    hlen        = len(train_haters)
    nhlen       = len(train_non_haters)
    htrain_len  = int(percentage * hlen)
    nhtrain_len = int(percentage * nhlen)

    #Creating Training List
    trainList = train_haters[0:htrain_len]
    trainList.extend(train_non_haters[0:nhtrain_len])
    #Creating Validation List
    valList = train_haters[htrain_len:]
    valList.extend(train_non_haters[nhtrain_len:])
    #Creating Testing DataPoint
    textList = list(test_haters)
    textList.extend(test_non_haters)
    
    train_mask = [0] * nodes
    test_mask  = [0] * nodes
    val_mask   = [0] * nodes
    
    for i in trainList:
      train_mask[graph_dict[i]] = 1;

    for i in valList:
      val_mask[graph_dict[i]] = 1;
    
    for i in textList:
      test_mask[graph_dict[i]] = 1;

    train_mask = torch.ByteTensor(train_mask)
    test_mask  = torch.ByteTensor(test_mask)
    val_mask   = torch.ByteTensor(val_mask)
    print("Splitting done")
    return train_mask, val_mask, test_mask



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



num_features = 100
num_classes = 2


print("GRAPHSAGE")
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SAGEConv(num_features, 32)
        self.conv2 = SAGEConv(32, 2)

    def forward(self):
        x= X
        #x = F.dropout(x, 0.5)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, 0.2)
        x = self.conv2(x, edge_index)
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
    for mask in [train_mask, val_mask, test_mask]:
        pred = logits[mask].max(1)[1]
        acc  = pred.eq(y[mask]).sum().item() / mask.sum().item()
        mfc  = f1_score(y[mask].detach().cpu(), pred.detach().cpu(), average='macro')
        accs.append(acc)
        Mf1_score.append(mfc)
    test_pred = logits[test_mask].max(1)[1]
    test_Metrics = evalMetric(y[test_mask].detach().cpu(), test_pred.detach().cpu())
    return accs, Mf1_score, test_Metrics



test_accs      = {}
test_mF1Score  = {}
test_f1Score   = {}
test_precision = {}
test_recall    = {}
test_roc       = {}
import copy

fin_macrof1Score= {}
fin_accs={}

val_macrof1Score= {}
val_accs={}

for percent in percentages:
    test_accs[percent]      = {}
    test_mF1Score[percent]  = {}
    test_f1Score[percent]   = {}
    test_precision[percent] = {}
    test_recall[percent]    = {}
    test_roc[percent]       = {}

    fin_macrof1Score[percent] = {}
    fin_accs[percent]         = {}
    val_macrof1Score[percent]  = {}
    val_accs[percent]          = {}

    for fold in validationFold:
        with open(dataDirectory+'GAB_Data/hateval'+fold+'.json') as json_file:
            test_haters = json.load(json_file)
        with open(dataDirectory+'GAB_Data/nonhateval'+fold+'.json') as json_file:
            test_non_haters = json.load(json_file)
        
        train_haters     = Diff(haters, test_haters) 
        train_non_haters = Diff(non_haters, test_non_haters)
        
        print(fold)
        print(percent, "Created features and labels and masking function")
       
        test_accs[percent][fold]      = []
        test_mF1Score[percent][fold]  = []
        test_f1Score[percent][fold]   = []
        test_precision[percent][fold] = []
        test_recall[percent][fold]    = []
        test_roc[percent][fold]       = []

        fin_macrof1Score[percent][fold]  = []
        fin_accs[percent][fold]          = []
        val_macrof1Score[percent][fold]  = []
        val_accs[percent][fold]          = []

         
        print("-------------", fold,"------------------------") 
        for i in range(0,5):
            torch.manual_seed(30+i)
            np.random.seed(30+i)
            X, y = getData()
            train_mask, val_mask, test_mask = ratio_split(percent, train_haters, train_non_haters, 
                                                                test_haters, test_non_haters, nodes)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model      = Net().to(device)
            edge_index = edge_index.to(device)
            use_gpu = torch.cuda.is_available()
            y = y.to(device)
            X = X.to(device)
            train_mask = train_mask.to(device)
            test_mask  = test_mask.to(device)
            val_mask   = val_mask.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            best_val_acc = best_MfScore = test_acc = test_mfscore = 0
            evalObject = None
            for epoch in range(1, 51):
                train()
                Accuracy, F1Score, test_Metrics = test()
                if Accuracy[1] > best_val_acc:
                    best_val_acc = Accuracy[1]
                    test_acc     = Accuracy[2]
                    best_MfScore = F1Score[1]
                    test_mfscore = F1Score[2]
                    evalObject = copy.deepcopy(test_Metrics)
                log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                # print(log.format(epoch, train_acc, val_acc, test_acc))
            test_accs[percent][fold].append(evalObject['accuracy'])
            test_mF1Score[percent][fold].append(evalObject['mF1Score'])
            test_f1Score[percent][fold].append(evalObject['f1Score'])
            test_precision[percent][fold].append(evalObject['precision'])
            test_recall[percent][fold].append(evalObject['recall'])
            test_roc[percent][fold].append(evalObject['auc'])
            val_macrof1Score[percent][fold].append(best_MfScore)
            val_accs[percent][fold].append(best_val_acc)



def getFoldWiseResult(dict_fold):
    FinalRes= {}
    for percent in percentages:
        FinalRes[percent]=[]
        for fold in validationFold:
            FinalRes[percent].extend(dict_fold[percent][fold])
    
    for percent in percentages:
        print(percent, '\t',np.mean(FinalRes[percent]), '\t', np.std(FinalRes[percent]))
    print("\n")



print("Test Accuracy:")
getFoldWiseResult(test_accs)
print("Test Macro-F1Score:")
getFoldWiseResult(test_mF1Score)
print("Test F1Score:")
getFoldWiseResult(test_f1Score)
print("Test Precision:")
getFoldWiseResult(test_precision)
print("Test Recall:")
getFoldWiseResult(test_recall)
print("Test Roc:")
getFoldWiseResult(test_roc)



#print("Validation Accuracy")
#ValFinalAcc= {}
#
#for percent in percentages:
#    print("Val:"+str(percent))
#    ValFinalAcc[percent]=[]
#    for zz in validationFold:
#        print(np.mean(val_accs[percent][zz]), np.std(val_accs[percent][zz]), val_accs[percent][zz])
#        ValFinalAcc[percent].extend(val_accs[percent][zz])
#
#print("\nValidation Macro F1 Score")
#ValFinalF1= {}
#
#for percent in percentages:
#    print("Validation F1-Score:"+str(percent))
#    ValFinalF1[percent] = []
#    for zz in validationFold:
#        print(np.mean(val_macrof1Score[percent][zz]), np.std(val_macrof1Score[percent][zz]), val_macrof1Score[percent][zz])
#        ValFinalF1[percent].extend(val_macrof1Score[percent][zz])
#
#print("\nTest Accuracy")
#TestFinalAcc= {}
#for percent in percentages:
#    print("Test:"+str(percent))
#    TestFinalAcc[percent]=[]
#    for zz in validationFold:
#        print(np.mean(test_accs[percent][zz]), np.std(test_accs[percent][zz]), test_accs[percent][zz])
#        TestFinalAcc[percent].extend(test_accs[percent][zz])
#
#print("\nTest Macro F1 Score")
#TestFinalF1= {}
#for percent in percentages:
#    print("Test F1-Score:"+str(percent))
#    TestFinalF1[percent]=[]
#    for zz in validationFold:
#        print(np.mean(test_macrof1Score[percent][zz]), np.std(test_macrof1Score[percent][zz]), test_macrof1Score[percent][zz])
#        TestFinalF1[percent].extend(test_macrof1Score[percent][zz])
##
#print("\nFinal Test Accuracy")
#endFinalAcc= {}
#for percent in percentages:
#    print("Fin Test:"+str(percent))
#    endFinalAcc[percent]=[]
#    for zz in validationFold:
#        print(np.mean(fin_accs[percent][zz]), np.std(fin_accs[percent][zz]), fin_accs[percent][zz])
#        endFinalAcc[percent].extend(fin_accs[percent][zz])
#
#print("\nFinal Test Macro F1 Score")
#endFinalF1= {}
#for percent in percentages:
#    print("Fin Test F1-Score:"+str(percent))
#    endFinalF1[percent]=[]
#    for zz in validationFold:
#        print(np.mean(fin_macrof1Score[percent][zz]), np.std(fin_macrof1Score[percent][zz]), fin_macrof1Score[percent][zz])
#        endFinalF1[percent].extend(fin_macrof1Score[percent][zz])



print("-----------------------------------------------------------------------------------------")
# print("\n--Validation--")
# for i in ValFinalAcc:
#     print(i, np.mean(ValFinalAcc[i]), np.std(ValFinalAcc[i]))
# for i in ValFinalF1:
#     print(i, np.mean(ValFinalF1[i]), np.std(ValFinalF1[i]))

# print("\n--Test--")
# print("Accuracy:")
# for i in TestFinalAcc:
#     print(i, np.mean(TestFinalAcc[i]), np.std(TestFinalAcc[i]))
# print("Macro F1-Score:")
# for i in TestFinalF1:
#     print(i, np.mean(TestFinalF1[i]), np.std(TestFinalF1[i]))


#print("\n--Fin--")
#print("Accuracy:")
#for i in endFinalAcc:
#    print(i, np.mean(endFinalAcc[i]), np.std(endFinalAcc[i]))
#print("Macro F1-Score:")
#for i in endFinalF1:
#    print(i, np.mean(endFinalF1[i]), np.std(endFinalF1[i]))
#
