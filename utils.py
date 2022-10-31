import numpy as np
import torch 
import torch.nn as nn

from itertools import combinations
from operator import itemgetter


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def create_edgeNode_relation(num_nodes, self_loops=False):
    if self_loops:
        indices = np.ones([num_nodes, num_nodes])
    else:
        indices = np.ones([num_nodes, num_nodes]) - np.eye(num_nodes)
    rel_rec = np.array(encode_onehot(np.where(indices)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(indices)[1]), dtype=np.float32)
    rel_rec = torch.from_numpy(rel_rec)
    rel_send = torch.from_numpy(rel_send)
    
    return rel_rec, rel_send


def edge_accuracy(preds, target):
    """compute pairwise group accuracy"""
    _, preds = preds.max(-1)
    correct = preds.float().data.eq(
        target.float().data.view_as(preds)).cpu().sum()
    return np.float(correct) / (target.size(0) * target.size(1))


def edge_precision(preds, target):
    """compute pairwise group/non-group recall"""
    _, preds = preds.max(-1)
    true_possitive = ((preds[target==1]==1).cpu().sum()).item()
    total_possitive = ((preds[preds==1]).cpu().sum()).item()
    if total_possitive==true_possitive:
        group_precision = 1
    true_negative = ((preds[target==0]==0).cpu().sum()).item()
    total_negative = ((preds[preds==0]==0).cpu().sum()).item()
    if total_negative==true_negative:
        non_group_precision = 1
    if total_possitive>0:
        group_precision = true_possitive/total_possitive
    if total_negative>0:
        non_group_precision = true_negative/total_negative
       
    #group_precision = ((target[preds==1]==1).cpu().sum())/preds[preds==1].cpu().sum()
    #non_group_precision = ((target[preds==0]==0).cpu().sum())/(preds[preds==0]==0).cpu().sum()
    return group_precision, non_group_precision


def edge_recall(preds, target):
    """compute pairwise group/non-group recall"""
    _,preds = preds.max(-1)
    retrived_possitive = ((preds[target==1]==1).cpu().sum()).item()
    total_possitive = ((target[target==1]).cpu().sum()).item()
    retrived_negative = ((preds[target==0]==0).cpu().sum()).item()
    total_negative = ((target[target==0]==0).cpu().sum()).item()
    
    if retrived_possitive==total_possitive:
        group_recall = 1
    if retrived_negative==total_negative:
        non_group_recall = 1
        
    if total_possitive > 0:
        group_recall = retrived_possitive/total_possitive
    if total_negative > 0:
        non_group_recall = retrived_negative/total_negative
    
    #group_recall = ((preds[target==1]==1).cpu().sum())/(target[target==1]).cpu().sum()
    #non_group_recall = ((preds[target==0]==0).cpu().sum())/(target[target==0]==0).cpu().sum()
    return group_recall, non_group_recall



"""Group Mitre """

def indices_to_clusters(l):
    """
    args:
        l: indices of clusters, e.g.. [0,0,1,1]
    return: clusters, e.g. [(0,1),(2,3)]
    """
    d = dict()
    for i,v in enumerate(l):
        d[v] = d.get(v,[])
        d[v].append(i)
    clusters = list(d.values())
    return clusters


def compute_mitre(target, predict):
    target_sets = [set(c) for c in target]
    predict_sets = [set(c) for c in predict]
    total_misses = 0.
    total_corrects = 0.
    size_predict = len(predict_sets)
    for cl in target_sets:
        size_cl = len(cl)
        total_corrects += size_cl-1
        if size_cl==1: # if the size of the cluster is 1, there are no missing links
            continue
        if True in [cl.issubset(cp) for cp in predict_sets]:
            # if the cluster is a subset of a cluster in the predicts clustering,
            # there are no missing links
            continue
        possible_misses = range(1, min(size_cl-1, size_predict-1)+1)
        for n_miss in possible_misses:
            indi_combs = list(combinations(range(size_predict), n_miss+1))
            possible_comb_sets = [set().union(*(itemgetter(*a)(predict_sets))) for a in indi_combs]
            if True in [cl.issubset(cp) for cp in possible_comb_sets]:
                total_misses+=n_miss
                break
                
    return (total_corrects-total_misses)/total_corrects




def create_counterPart(a):
    """
    add fake counter parts for each agent
    args:
      a: list of groups; e.g. a=[[0,1],[2],[3,4]]
    """
    a_p = []
    for group in a:
        if len(group)==1:#singleton
            element = group[0]
            element_counter = -(element+1)#assume element is non-negative
            new_group = [element, element_counter]
            a_p.append(new_group)
        else:
            a_p.append(group)
            for element in group:
                element_counter = -(element+1)
                a_p.append([element_counter])
    return a_p



def compute_groupMitre(target, predict):
    """
    compute group mitre
    args: 
      target,predict: list of groups; [[0,1],[2],[3,4]]
    return: recall, precision, F1
    """
    #create fake counter agents
    target_p = create_counterPart(target)
    predict_p = create_counterPart(predict)
    recall = compute_mitre(target_p, predict_p)
    precision = compute_mitre(predict_p, target_p)
    if recall==0 or precision==0:
        F1 = 0
    else:
        F1 = 2*recall*precision/(recall+precision)
    return recall, precision, F1


def compute_gmitre_loss(target, predict):
    _,_, F1 = compute_groupMitre(target, predict)
    return 1-F1



def compute_groupMitre_labels(target, predict):
    """
    compute group mitre given indices
    args: target, predict: list of indices of groups
       e.g. [0,0,1,1]
    """
    target = indices_to_clusters(target)
    predict = indices_to_clusters(predict)
    recall, precision, F1 = compute_groupMitre(target, predict)
    return recall, precision, F1

