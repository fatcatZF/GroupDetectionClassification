from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
import datetime

import torch.optim as optim
from torch.optim import lr_scheduler

from utils import *
from models import *

import networkx as nx 
from networkx import connected_components


parser = argparse.ArgumentParser()
parser.add_argument("--no-cuda", action="store_true", default=False, 
                    help="Disables CUDA training.")
parser.add_argument("--epochs", type=int, default=200, 
                    help="Number of epochs to train.")
parser.add_argument("--batch-size", type=int, default=128,
                    help="Number of samples per batch.")
parser.add_argument("--lr", type=float, default=0.005,
                    help="Initial learning rate.")
parser.add_argument("--encoder-hidden", type=int, default=128,
                    help="Number of hidden units.")
parser.add_argument("--suffix", type=str, default="zara01",
                    help="Suffix for training data ")
parser.add_argument("--split", type=str, default="split00",
                    help="Split of the dataset.")
parser.add_argument("--encoder-dropout", type=float, default=0.3,
                    help="Dropout rate (1-keep probability).")
parser.add_argument("--save-folder", type=str, default="logs/class",
                    help="Where to save the trained model, leave empty to not save anything.")
parser.add_argument("--load-folder", type=str, default='', 
                    help="Where to load the trained model.")
parser.add_argument("--edge-types", type=int, default=2,
                    help="The number of edge types to infer.")
parser.add_argument("--dims", type=int, default=2,
                    help="The number of feature dimensions.")
parser.add_argument("--edge-dim", type=int, default=8, 
                    help="The number of temporal edge feature extractors.")
parser.add_argument("--kernel-size", type=int, default=5,
                    help="Kernel size of WavenetNRI Encoder")

parser.add_argument("--depth", type=int, default=1,
                    help="depth of Wavenet CNN res blocks.")

parser.add_argument("--timesteps", type=int, default=15,
                    help="The number of time steps per sample.")
parser.add_argument("--lr-decay", type=int, default=100,
                    help="After how epochs to decay LR factor of gamma.")
parser.add_argument("--gamma", type=float, default=0.5,
                    help="LR decay factor.")

parser.add_argument("--group-weight", type=float, default=0.5,
                    help="group weight.")
parser.add_argument("--ng-weight", type=float, default=0.5,
                    help="Non-group weight.")

parser.add_argument("--grecall-weight", type=float, default=0.65,
                    help="group recall.")


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)


log = None
#Save model and meta-data
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = "{}/{}_{}/".format(args.save_folder,timestamp, args.suffix+args.split)
    os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, "metadata.pkl")
    encoder_file = os.path.join(save_folder, "encoder.pt")
    
    log_file = os.path.join(save_folder, "log.txt")
    log = open(log_file, 'w')
    pickle.dump({"args":args}, open(meta_file, 'wb'))
    
else:
    print("WARNING: No save_folder provided!"+
          "Testing (within this script) will throw an error.")

#Load data
data_folder = os.path.join("data/pedestrian/", args.suffix)
data_folder = os.path.join(data_folder, args.split)


with open(os.path.join(data_folder, "tensors_train.pkl"), 'rb') as f:
    examples_train = pickle.load(f)
with open(os.path.join(data_folder, "labels_train.pkl"), 'rb') as f:
    labels_train = pickle.load(f)
with open(os.path.join(data_folder, "tensors_valid.pkl"), 'rb') as f:
    examples_valid = pickle.load(f)
with open(os.path.join(data_folder, "labels_valid.pkl"), 'rb') as f:
    labels_valid = pickle.load(f)
with open(os.path.join(data_folder, "tensors_test.pkl"),'rb') as f:
    examples_test = pickle.load(f)
with open(os.path.join(data_folder, "labels_test.pkl"), 'rb') as f:
    labels_test = pickle.load(f)


encoder = EncoderSym(args.dims, args.edge_dim, args.encoder_hidden, args.edge_types,
                        args.kernel_size, args.depth)


cross_entropy_weight = torch.tensor([args.ng_weight, args.group_weight]) 


if args.load_folder:
    encoder_file = os.path.join(args.load_folder, "encoder.pt")
    encoder.load_state_dict(torch.load(encoder_file))
    args.save_folder = False


if args.cuda:
    encoder.cuda()
    cross_entropy_weight = cross_entropy_weight.cuda()


optimizer = optim.SGD(list(encoder.parameters()), lr=args.lr, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)


def train(epoch, best_F1):
    t = time.time()
    loss_train = []
    recall_train = []
    precision_train = []
    F1_train = []
    loss_val = []
    recall_val = []
    precision_val = []
    F1_val = []

    encoder.train()

    training_indices = np.arange(len(examples_train))
    np.random.shuffle(training_indices)

    optimizer.zero_grad()
    idx_count = 0
    accumulation_steps = min(args.batch_size, len(examples_train)) #Initialization of accumulation steps

    for idx in training_indices:
        example = examples_train[idx]
        label = labels_train[idx]
        #add batch dimension
        example = example.unsqueeze(0)
        label = label.unsqueeze(0)
        num_atoms = example.size(1) #get number of atoms
        rel_rec, rel_send = create_edgeNode_relation(num_atoms, self_loops=False)

        if args.cuda:
            example = example.cuda()
            label = label.cuda()
            rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()

        example = example.float()
        logits = encoder(example, rel_rec, rel_send)
        #shape: [n_batch=1, n_edges, n_edgetypes=2]
        z = F.softmax(logits, dim=-1)
        #shape: [n_batch=1, n_edges, n_edgetypes=2]
        z = z[:,:,1]
        #shape: [n_batch=1, n_edges]

        output = logits.view(logits.size(0)*logits.size(1),-1)
        target = label.view(-1)
        
        loss = F.cross_entropy(output, target.long(), weight=cross_entropy_weight)

        loss_train.append(loss.item())
        loss = loss/accumulation_steps #average by dividing accumulation steps
        loss.backward()
        idx_count+=1

        if idx_count%args.batch_size==0 or idx_count==len(examples_train):
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accumulation_steps = min(args.batch_size, len(examples_train)-idx_count)

        sims_label = label2mat(label.flatten(), num_atoms)
        sims_label_G = nx.from_numpy_array(sims_label.astype("int"))
        clusters_label = list(connected_components(sims_label_G))
        clusters_label = [list(c) for c in clusters_label]

        sims_predict = label2mat(z.flatten(), num_atoms)
        #shape: [n_atoms, n_atoms]
        sims_predict = (sims_predict>0.5).astype("int")
        sims_predict_G = nx.from_numpy_array(sims_predict)
        clusters_predict = list(connected_components(sims_predict_G))
        clusters_predict = [list(c) for c in clusters_predict]

        recall, precision, F1 = compute_groupMitre(clusters_label, clusters_predict)
        recall_train.append(recall)
        precision_train.append(precision)
        F1_train.append(F1)



    encoder.eval()
    valid_indices = np.arange(len(examples_valid))

    with torch.no_grad():
        for idx in valid_indices:
            example = examples_valid[idx]
            label = labels_valid[idx]
            example = example.unsqueeze(0)
            label = label.unsqueeze(0)
            num_atoms = example.size(1)
            rel_rec, rel_send = create_edgeNode_relation(num_atoms, self_loops=False)
            if args.cuda:
                example = example.cuda()
                label = label.cuda()
                rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
            example = example.float()
            logits = encoder(example, rel_rec, rel_send)
            z = F.softmax(logits, dim=-1)
            #shape: [n_batch=1, n_edges, n_edgetypes=2]
            z = z[:,:,1]
            #shape: [n_batch=1, n_edges]
            output = logits.view(logits.size(0)*logits.size(1),-1)
            target = label.view(-1)
            loss_current = F.cross_entropy(output, target.long(), weight=cross_entropy_weight)

            #move tensors back to cpu
            example = example.cpu()
            rel_rec, rel_send = rel_rec.cpu(), rel_send.cpu()
            
            loss_val.append(loss_current.item())

            sims_label = label2mat(label.flatten(), num_atoms)
            sims_label_G = nx.from_numpy_array(sims_label.astype("int"))
            clusters_label = list(connected_components(sims_label_G))
            clusters_label = [list(c) for c in clusters_label]

            sims_predict = label2mat(z.flatten(), num_atoms)
            #shape: [n_atoms, n_atoms]
            sims_predict = (sims_predict>0.5).astype("int")
            sims_predict_G = nx.from_numpy_array(sims_predict)
            clusters_predict = list(connected_components(sims_predict_G))
            clusters_predict = [list(c) for c in clusters_predict]

            recall, precision, F1 = compute_groupMitre(clusters_label, clusters_predict)
            recall_val.append(recall)
            precision_val.append(precision)
            F1_val.append(F1)

    
    print("Epoch: {:04d}".format(epoch),
          "loss_train: {:.10f}".format(np.mean(loss_train)),
          "recall_train: {:.10f}".format(np.mean(recall_train)),
          "precision_train: {:.10f}".format(np.mean(precision_train)),
          "F1_train: {:.10f}".format(np.mean(F1_train)),
          "loss_val: {:.10f}".format(np.mean(loss_val)),
          "recall_val: {:.10f}".format(np.mean(recall_val)),
          "precision_val: {:.10f}".format(np.mean(precision_val)),
          "F1_val: {:.10f}".format(np.mean(F1_val)))
    if args.save_folder and np.mean(F1_val) > best_F1:
        torch.save(encoder, encoder_file)
        print("Best model so far, saving...")
        print("Epoch: {:04d}".format(epoch),
          "loss_train: {:.10f}".format(np.mean(loss_train)),
          "recall_train: {:.10f}".format(np.mean(recall_train)),
          "precision_train: {:.10f}".format(np.mean(precision_train)),
          "F1_train: {:.10f}".format(np.mean(F1_train)),
          "loss_val: {:.10f}".format(np.mean(loss_val)),
          "recall_val: {:.10f}".format(np.mean(recall_val)),
          "precision_val: {:.10f}".format(np.mean(precision_val)),
          "F1_val: {:.10f}".format(np.mean(F1_val)), file=log)
        log.flush()

    
    return np.mean(F1_val)




def test():
    t = time.time()
    recall_test = []
    precision_test = []
    F1_test = []
    
    encoder = torch.load(encoder_file)
    encoder.eval()
    test_indices = np.arange(len(examples_test))

    with torch.no_grad():
        for idx in test_indices:
            example = examples_test[idx]
            label = labels_test[idx]
            example = example.unsqueeze(0)
            label = label.unsqueeze(0)
            num_atoms = example.size(1) #get number of atoms
            rel_rec, rel_send = create_edgeNode_relation(num_atoms, self_loops=False)
            if args.cuda:
                example = example.cuda()
                label = label.cuda()
                rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
            example = example.float()
            logits = encoder(example, rel_rec, rel_send)
            z = F.softmax(logits, dim=-1)
            #shape: [n_batch=1, n_edges, n_edgetypes=2]
            z = z[:,:,1]
            #shape: [n_batch=1, n_edges]
            
            sims_label = label2mat(label.flatten(), num_atoms)
            sims_label_G = nx.from_numpy_array(sims_label.astype("int"))
            clusters_label = list(connected_components(sims_label_G))
            clusters_label = [list(c) for c in clusters_label]

            sims_predict = label2mat(z.flatten(), num_atoms)
            #shape: [n_atoms, n_atoms]
            sims_predict = (sims_predict>0.5).astype("int")
            sims_predict_G = nx.from_numpy_array(sims_predict)
            clusters_predict = list(connected_components(sims_predict_G))
            clusters_predict = [list(c) for c in clusters_predict]

            recall, precision, F1 = compute_groupMitre(clusters_label, clusters_predict)
            recall_test.append(recall)
            precision_test.append(precision)
            F1_test.append(F1)


    
    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    print(
          "recall_train: {:.10f}".format(np.mean(recall_test)),
          "precision_train: {:.10f}".format(np.mean(precision_test)),
          "F1_train: {:.10f}".format(np.mean(F1_test)))



        
        

#Train model

t_total = time.time()
best_F1 = 0.
best_epoch = 0

for epoch in range(args.epochs):
    val_F1 = train(epoch, best_F1)
    if val_F1 > best_F1:
        best_F1 = val_F1
        best_epoch = epoch
        
print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch))

test()
