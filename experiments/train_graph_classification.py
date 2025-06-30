# -*- coding: utf-8 -*-
import copy
import sys
sys.path.append("..")
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric import datasets
from torch_geometric.datasets import TUDataset
import torch_geometric.utils as utils
from sat.models import GraphTransformer
from sat.data import GraphDataset
from sat.utils import count_parameters
from sat.position_encoding import POSENCODINGS
from sat.gnn_layers import GNN_TYPES
from timeit import default_timer as timer
from groupvit.models import GroupGraphTransformer, GraphViT
from model.simclr import simclr
from infonce import InfoNCE
from experiments.arguments import load_args
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from groupvit.plot_group_tokens import plot_batch_graphs, plot_batch_graphs_all_layers

def evaluate_classification(y_true, y_pred, metric='acc'):
    y_true = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred

    if metric == 'acc':
        return accuracy_score(y_true, y_pred)
    elif metric == 'f1':
        return f1_score(y_true, y_pred, average='macro')
    elif metric == 'auc':
        return roc_auc_score(y_true, y_pred)  # only valid if y_pred are probs
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    

def train_epoch(model, model_simclr, loader, criterion, optimizer, lr_scheduler, epoch, use_cuda=False, unsupervised=False):
    model.train()
    model_simclr.train()

    running_loss = 0.0

    tic = timer()
    for i, data in enumerate(loader):
        #print(data)
        size = len(data.y)
        if epoch < args.warmup:
            iteration = epoch * len(loader) + i
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_scheduler(iteration)
        if args.abs_pe == 'lap':
            # sign flip as in Bresson et al. for laplacian PE
            sign_flip = torch.rand(data.abs_pe.shape[-1])
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            data.abs_pe = data.abs_pe * sign_flip.unsqueeze(0)

        if use_cuda:
            data = data.cuda()

        optimizer.zero_grad()
        # print("train: \n")
        output, attn_dict_list = model(data, return_attn=True)
        soft_list = [d["soft"] for d in attn_dict_list if d is not None and "soft" in d]

        # print("soft_list:", soft_list)
        # output = model(data)

        # get features from simclr
        # features_simclr = model_simclr(data)
        # features_simclr = model_simclr(data.x, data.edge_index, data.batch)
        # print("features_simclr shape:", features_simclr.shape)  # 確保 features_simclr 的形狀是 [batch_size, embedding_dim]
        # print("output shape:", output.shape)  # 確保 x 的形狀是 [num_nodes, num_features]
        # loss = criterion(output, features_simclr)
        if unsupervised:
            features_simclr = model_simclr(data)
            loss = criterion(output, features_simclr)
        else:
            loss = criterion(output, data.y.squeeze())
            # loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * size

    toc = timer()
    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    print('Train loss: {:.4f} time: {:.2f}s'.format(
          epoch_loss, toc - tic))
    return epoch_loss


def eval_epoch(model, model_simclr, loader, criterion, use_cuda=False, split='Val', unsupervised=False):
    model.eval()
    model_simclr.eval()

    running_loss = 0.0
    y_pred = []
    y_true = []

    tic = timer()
    with torch.no_grad():
        for data in loader:
            size = len(data.y)
            if use_cuda:
                data = data.cuda()

            # print("evaluation \n")
            output, attn_dict_list = model(data, return_attn=True)
            # print("attn_dict_list shape:", len(attn_dict_list))
            # print("attn_dict_list[0] = ", attn_dict_list[0])
            plot_batch_graphs_all_layers(data, attn_dict_list)
            # plot_batch_graphs(data, attn_dict_list)
            # soft_list = [d["soft"] for d in attn_dict_list if d is not None and "soft" in d]
            # hard_list = [d["hard"] for d in attn_dict_list if d is not None and "hard" in d]
            # print("soft_list:", soft_list)
            # print("hard_list:", hard_list)
            # print("hard_list shape:", [h.shape for h in hard_list])
            # print("hard_list size:", len(hard_list))
            # plot_single_graph(data=data, attn_dict=hard_list[-1])
            # print("batch = ", attn_dict_list[0]['hard'].shape[0])
            # print("attn_dict_list[0] hard shape out = ", attn_dict_list[0]['hard'].shape)
            # print("attn_dict_list[1] hard shape out = ", attn_dict_list[1]['hard'].shape)
            # for i in range(attn_dict_list[0].get("hard", 0).shape[0]):
            #     hard_list = attn_dict_list[0].get("hard")
            #     # print("hard_list shape:", [h.shape for h in hard_list])
            #     hard_i = hard_list[i] if hard_list is not None else None
            #     # print("hard_i shape:", hard_i.shape)
            #     print("hard_i = ", hard_i)
            #     hard_i_trimmed = hard_i[:, :, :data[i].num_nodes]
            #     plot_single_graph(data=data[i], attn_dict=hard_i_trimmed, title=f"{split} hard_list {i}")

            # get features from simclr
            # features_simclr = model_simclr(data)
            # features_simclr = model_simclr(data.x, data.edge_index, data.batch)
            # loss = criterion(output, features_simclr)
            if unsupervised:
                features_simclr = model_simclr(data)
                loss = criterion(output, features_simclr)
            else:
                loss = criterion(output, data.y.squeeze())
                # loss = criterion(output, data.y)
            # loss = criterion(output, data.y)
            y_true.append(data.y.cpu())
            y_pred.append(output.argmax(dim=-1).view(-1, 1).cpu())

            running_loss += loss.item() * size
    toc = timer()

    y_pred = torch.cat(y_pred).cpu()
    y_true = torch.cat(y_true).cpu()

    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    score = evaluate_classification(y_true, y_pred)

    print('{} loss: {:.4f} score: {:.4f} time: {:.2f}s'.format(
        split, epoch_loss, score, toc - tic))
    return score, epoch_loss


def infer_num_node_features(dataset):
    for data in dataset:
        if data.x is not None:
            return data.x.size(1)
    raise ValueError("All graphs in dataset have no node features.")

def main():
    global args
    args = load_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)
    data_path = args.data_path
    train_dataset = TUDataset(root=data_path, name=args.DS)
    input_size = infer_num_node_features(train_dataset)

    # train_dset = GraphDataset(datasets.ZINC(data_path, subset=True,
    #     split='train'), degree=True, k_hop=args.k_hop, se=args.se,
    #     use_subgraph_edge_attr=args.use_edge_attr)
    train_dset = GraphDataset(train_dataset, degree=True, k_hop=args.k_hop, se=args.se,
        use_subgraph_edge_attr=args.use_edge_attr)
    train_loader = DataLoader(train_dset, batch_size=args.batch_size,
            shuffle=True)

    print(train_dset[0])

    # val_dset = GraphDataset(datasets.ZINC(data_path, subset=True,
    #     split='val'), degree=True, k_hop=args.k_hop, se=args.se,
    #     use_subgraph_edge_attr=args.use_edge_attr)

    val_dataset = TUDataset(root=data_path, name=args.DS)
    val_dset = GraphDataset(val_dataset, degree=True, k_hop=args.k_hop, se=args.se,
        use_subgraph_edge_attr=args.use_edge_attr)    
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False)

    abs_pe_encoder = None
    if args.abs_pe and args.abs_pe_dim > 0:
        abs_pe_method = POSENCODINGS[args.abs_pe]
        abs_pe_encoder = abs_pe_method(args.abs_pe_dim, normalization='sym')
        if abs_pe_encoder is not None:
            abs_pe_encoder.apply_to(train_dset)
            abs_pe_encoder.apply_to(val_dset)

    # deg = torch.cat([
    #     utils.degree(data.edge_index[1], num_nodes=data.num_nodes) for
    #     data in train_dset])
    
    if args.model == "groupvit":
        pass
        # model = GroupGraphTransformer(in_size=input_size,
        #                      num_class=1,
        #                      d_model=args.dim_hidden,
        #                      dim_feedforward=2*args.dim_hidden,
        #                      dropout=args.dropout,
        #                      num_heads=args.num_heads,
        #                      num_layers=args.num_layers,
        #                      batch_norm=args.batch_norm,
        #                      abs_pe=args.abs_pe,
        #                      abs_pe_dim=args.abs_pe_dim,
        #                      gnn_type=args.gnn_type,
        #                      use_edge_attr=args.use_edge_attr,
        #                      num_edge_features=num_edge_features,
        #                      edge_dim=args.edge_dim,
        #                      k_hop=args.k_hop,
        #                      se=args.se,
        #                      deg=deg,
        #                      global_pool=args.global_pool)
        print("GroupGraphTransformer")
    elif args.model == "sat":
        pass
        # model = GraphTransformer(in_size=input_size,
        #                      num_class=1,
        #                      d_model=args.dim_hidden,
        #                      dim_feedforward=2*args.dim_hidden,
        #                      dropout=args.dropout,
        #                      num_heads=args.num_heads,
        #                      num_layers=args.num_layers,
        #                      batch_norm=args.batch_norm,
        #                      abs_pe=args.abs_pe,
        #                      abs_pe_dim=args.abs_pe_dim,
        #                      gnn_type=args.gnn_type,
        #                      use_edge_attr=args.use_edge_attr,
        #                      num_edge_features=num_edge_features,
        #                      edge_dim=args.edge_dim,
        #                      k_hop=args.k_hop,
        #                      se=args.se,
        #                      deg=deg,
        #                      global_pool=args.global_pool) 
    elif args.model == "graphvit":
        model = GraphViT(in_size=input_size,
                             num_class=2,
                             d_model=args.dim_hidden,
                            #  dim_feedforward=2*args.dim_hidden,
                            #  dropout=args.dropout,
                            #  num_heads=args.num_heads,
                            #  num_layers=args.num_layers,
                            #  batch_norm=args.batch_norm,
                             abs_pe=args.abs_pe,
                             abs_pe_dim=args.abs_pe_dim,
                             in_embed=False,
                             subgraph_embed=args.subgraph_embed,
                             num_group_tokens=[8, 4, 0],
                             num_output_groups=[8, 4],
                             embed_factors=[1, 1, 1], 
                             depths=[3, 2, 1],
                            #  gnn_type=args.gnn_type,
                            #  use_edge_attr=args.use_edge_attr,
                            #  num_edge_features=num_edge_features,
                            #  edge_dim=args.edge_dim,
                            #  k_hop=args.k_hop,
                            #  se=args.se,
                            #  deg=deg
                            )
        print("GraphViT")
    else:
        raise ValueError("Model type not supported")
    
    model_simclr = simclr(d_model=args.dim_hidden, num_gc_layers=args.num_layers)
    # model_simclr = simclr(in_size=input_size, d_model=args.dim_hidden, num_gc_layers=args.num_layers)

    if args.use_cuda:
        model.cuda()
        model_simclr.cuda()
    print("Total number of parameters: {}".format(count_parameters(model)))

    if args.unsupervised:
        # criterion = nn.MSELoss()
        criterion = InfoNCE(temperature=0.5)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - args.warmup)

    # lr_steps = args.lr / (args.warmup * len(train_loader))
    # def warmup_lr_scheduler(s):
    #     lr = s * lr_steps
    #     return lr

    if args.warmup is None:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                            factor=0.5,
                                                            patience=15,
                                                            min_lr=1e-05,
                                                            verbose=False)
    else:
        lr_steps = (args.lr - 1e-4) / args.warmup
        decay_factor = args.lr * args.warmup ** .5
        def lr_scheduler(s):
            if s < args.warmup:
                lr = 1e-4 + s * lr_steps
            else:
                lr = decay_factor * s ** -.5
            return lr

    # test_dset = GraphDataset(datasets.ZINC(data_path, subset=True,
    #     split='test'), degree=True, k_hop=args.k_hop, se=args.se,
    #     use_subgraph_edge_attr=args.use_edge_attr)

    test_dataset = TUDataset(root=data_path, name=args.DS)
    test_dset = GraphDataset(test_dataset, degree=True, k_hop=args.k_hop, se=args.se,
        use_subgraph_edge_attr=args.use_edge_attr)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False)
    
    #FIXME
    if abs_pe_encoder is not None:
        abs_pe_encoder.apply_to(test_dset)

    print("Training...")
    best_val_loss = float('inf')
    best_val_score = 0
    best_model = None
    best_epoch = 0
    logs = defaultdict(list)
    start_time = timer()
    for epoch in range(args.epochs):
        print("Epoch {}/{}, LR {:.6f}".format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        train_loss = train_epoch(model, model_simclr, train_loader, criterion, optimizer, lr_scheduler, epoch, args.use_cuda, unsupervised=args.unsupervised)
        # train_loss = train_epoch(model, model_simclr, train_loader, criterion, optimizer, warmup_lr_scheduler, epoch, args.use_cuda, unsupervised=args.unsupervised)
        val_score, val_loss = eval_epoch(model, model_simclr, val_loader, criterion, args.use_cuda, split='Val', unsupervised=args.unsupervised)
        test_score, test_loss = eval_epoch(model, model_simclr, test_loader, criterion, args.use_cuda, split='Test', unsupervised=args.unsupervised)

        if epoch >= args.warmup:
            lr_scheduler.step()

        logs['train_loss'].append(train_loss)
        logs['val_score'].append(val_score)
        logs['test_score'].append(test_score)
        if val_score > best_val_score:
            best_val_score = val_score
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())

    total_time = timer() - start_time
    print("best epoch: {} best val score: {:.4f}".format(best_epoch, best_val_score))
    model.load_state_dict(best_weights)

    print("Testing...")
    test_score, test_loss = eval_epoch(model, test_loader, criterion, args.use_cuda, split='Test')

    print("test ACC {:.4f}".format(test_score))

    if args.save_logs:
        logs = pd.DataFrame.from_dict(logs)
        logs.to_csv(args.outdir + '/logs.csv')
        results = {
            'test_score': test_score,
            'test_loss': test_loss,
            'val_score': best_val_score,
            'val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'total_time': total_time,
        }
        results = pd.DataFrame.from_dict(results, orient='index')
        results.to_csv(args.outdir + '/results.csv',
                       header=['value'], index_label='name')
        torch.save(
            {'args': args,
            'state_dict': best_weights},
            args.outdir + '/model.pth')


if __name__ == "__main__":
    main()
