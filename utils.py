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


def label2mat(label, n_atoms):
    """
    convert edge labels to matrix
    args:
      label, shape: [n_edges]
      n_atoms: the number of atoms
    """
    rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
    rel_rec, rel_send = rel_rec.float(), rel_send.float()
    label = torch.diag_embed(label) #shape: [n_edges, n_edges]
    label = label.float()
    label_converted = torch.matmul(rel_send.t(), 
                                           torch.matmul(label, rel_rec))
    sims = label_converted.cpu().detach().numpy()
    #shape: [n_atoms, n_atoms]
    sims = 0.5*(sims+sims.T)
    #to make the matrix symmetric
    return sims




"""Correlation Clustering"""
def compute_all_clusterings(indices):
    """
    args:
        indices: indices of items
    """
    if len(indices)==1:
        yield [indices]
        return
    first = indices[0]
    for smaller in compute_all_clusterings(indices[1:]):
        # insert "first" in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n]+[[first]+subset]+smaller[n+1:]
        yield [[first]]+smaller


def compute_clustering_score(sims, clustering):
    """
    args:
        sims: similarity matrix
        clustering: list of lists denoting clusters
    """
    score = 0.
    for cluster in clustering:
        if len(cluster)>=2:
            combs = list(combinations(cluster, 2))
            for comb in combs:
                score += sims[comb]
    return score


def merge_2_clusters(current_clustering, indices):
    """
    merge 2 clusters of current clustering
    args:
        current_clustering: list of lists denoting clusters
        indices(tuple): indices of 2 clusters of current clustering
    """
    assert len(current_clustering)>1
    num_clusters = len(current_clustering)
    cluster1 = current_clustering[indices[0]]
    cluster2 = current_clustering[indices[1]]
    merged_cluster = cluster1+cluster2
    new_clustering = [merged_cluster]
    for i in range(num_clusters):
        if i!=indices[0] and i!=indices[1]:
            new_clustering.append(current_clustering[i])
    return new_clustering


def greedy_approximate_best_clustering(sims):
    """
    args:
        sims(numpy ndarray): similarity matrices, shape:[n_atoms, n_atoms]
        current_clustering: a list of lists denoting clusters
        current_score: current clustering score
    """
    num_atoms = sims.shape[0]
    current_cluster_indices = list(range(num_atoms))
    current_clustering = [[i] for i in current_cluster_indices]
    current_score = 0.
    merge_2_indices = list(combinations(current_cluster_indices, 2))
    best_clustering = current_clustering
    
    
    while(True):
        #merge 2 clusters hierachically
        
        #if len(current_clustering)==1: #cannot be merged anymore
        #    return current_clustering, current_score
        
        best_delta = 0
        for merge_index in merge_2_indices:
            new_clustering = merge_2_clusters(current_clustering, merge_index)
            new_score = compute_clustering_score(sims, new_clustering)
            delta = new_score-current_score
            if delta>best_delta:
                best_clustering = new_clustering
                best_delta = delta
                current_score = new_score
        if best_delta<=0:
            return best_clustering, current_score
        
        current_clustering = best_clustering
        current_num_clusters = len(current_clustering)
        if current_num_clusters==1:
            return current_clustering, current_score
        cluster_indices = list(range(current_num_clusters))
        merge_2_indices = list(combinations(cluster_indices, 2))

