# -*- coding: utf-8 -*-
import os
import copy
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric import datasets
import torch_geometric.utils as utils
from sat.models import GraphTransformer
from sat.data import GraphDataset
from sat.utils import count_parameters
from sat.position_encoding import POSENCODINGS
from sat.gnn_layers import GNN_TYPES
from sat.utils import add_zeros, extract_node_feature
from timeit import default_timer as timer
from groupvit.models import GraphViT
from experiments.arguments import load_args

from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred import Evaluator
from groupvit.plot_group_tokens import plot_batch_graphs_all_layers, plot_batch_graphs


def train_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, use_cuda=False):
    model.train()

    running_loss = 0.0

    tic = timer()
    for i, data in enumerate(loader):
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
        output = model(data)

        loss = criterion(output, data.y.squeeze())
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * size

    toc = timer()
    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    print('Train loss: {:.4f} time: {:.2f}s'.format(
          epoch_loss, toc - tic))
    return epoch_loss


def eval_epoch(model, loader, criterion, use_cuda=False, split='Val', get_attn=False):
    model.eval()

    running_loss = 0.0
    y_pred = []
    y_true = []

    tic = timer()
    with torch.no_grad():
        for data in loader:
            size = len(data.y)
            if use_cuda:
                data = data.cuda()

            output, attn_dict_list = model(data, return_attn=True)
            if args.plot_attn:
                plot_batch_graphs_all_layers(data, attn_dict_list, attn_type='soft')
                # plot_batch_graphs_all_layers(data, attn_dict_list, attn_type='hard')
                # plot_batch_graphs(data, attn_dict_list)
            loss = criterion(output, data.y.squeeze())
            
            y_true.append(data.y.cpu())
            y_pred.append(output.argmax(dim=-1).view(-1, 1).cpu())

            running_loss += loss.item() * size

    toc = timer()
    y_pred = torch.cat(y_pred).numpy()
    y_true = torch.cat(y_true).numpy()

    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    evaluator = Evaluator(name=args.dataset)
    score = evaluator.eval({'y_pred': y_pred,
                         'y_true': y_true})['acc']
    print('{} loss: {:.4f} score: {:.4f} time: {:.2f}s'.format(
          split, epoch_loss, score, toc - tic))
    return score, epoch_loss


def main():
    global args
    args = load_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)
    data_path = '../datasets'
    num_edge_features = 7

    if args.not_extract_node_feature:
        transform = add_zeros
        input_size = 1
    else:
        from functools import partial
        transform = partial(extract_node_feature, reduce=args.aggr)
        input_size = num_edge_features

    dataset = PygGraphPropPredDataset(name=args.dataset, root=data_path,
                                      transform=transform)
    split_idx = dataset.get_idx_split()

    train_dset = GraphDataset(dataset[split_idx['train']], degree=True,
        k_hop=args.k_hop, se=args.se, use_subgraph_edge_attr=args.use_edge_attr,
        return_complete_index=False)

    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
    print(len(train_dset))
    print(train_dset[0])

    val_dset = GraphDataset(dataset[split_idx['valid']], degree=True,
        k_hop=args.k_hop, se=args.se, use_subgraph_edge_attr=args.use_edge_attr,
        return_complete_index=False)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False)

    abs_pe_encoder = None
    if args.abs_pe and args.abs_pe_dim > 0:
        abs_pe_method = POSENCODINGS[args.abs_pe]
        abs_pe_encoder = abs_pe_method(args.abs_pe_dim, normalization='sym')
        if abs_pe_encoder is not None:
            abs_pe_encoder.apply_to(train_dset)
            abs_pe_encoder.apply_to(val_dset)

    if 'pna' in args.gnn_type or args.gnn_type == 'mpnn':
        deg = torch.cat([
            utils.degree(data.edge_index[1], num_nodes=data.num_nodes) for data in train_dset])
    else:
        deg = None
    print(deg)

    if args.model == 'sat':
        model = GraphTransformer(in_size=input_size,
                                num_class=dataset.num_classes,
                                d_model=args.dim_hidden,
                                dim_feedforward=2*args.dim_hidden,
                                dropout=args.dropout,
                                num_heads=args.num_heads,
                                num_layers=args.num_layers,
                                batch_norm=args.batch_norm,
                                abs_pe=args.abs_pe,
                                abs_pe_dim=args.abs_pe_dim,
                                gnn_type=args.gnn_type,
                                k_hop=args.k_hop,
                                use_edge_attr=args.use_edge_attr,
                                num_edge_features=num_edge_features,
                                edge_dim=args.edge_dim,
                                se=args.se,
                                deg=deg,
                                in_embed=False,
                                edge_embed=False,
                                global_pool=args.global_pool)
    elif args.model == "graphvit":
        model = GraphViT(in_size=input_size,
                             num_class=dataset.num_classes,
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
                             num_group_tokens=[64, 8, 0],
                             num_output_groups=[64, 8],
                             embed_factors=[1, 1, 1], 
                             depths=[3, 2, 1],
                             gumbel_assignment=args.gumbel_assignment,
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
        raise ValueError("Unknown model type: {}".format(args.model))

    if args.use_cuda:
        model.cuda()
    print("Total number of parameters: {}".format(count_parameters(model)))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - args.warmup)

    lr_steps = args.lr / (args.warmup * len(train_loader))
    def warmup_lr_scheduler(s):
        lr = s * lr_steps
        return lr

    test_dset = GraphDataset(dataset[split_idx['test']], degree=True,
        k_hop=args.k_hop, se=args.se, use_subgraph_edge_attr=args.use_edge_attr,
        return_complete_index=False)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False)

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
        train_loss = train_epoch(model, train_loader, criterion, optimizer, warmup_lr_scheduler, epoch, args.use_cuda)
        val_score, val_loss = eval_epoch(model, val_loader, criterion, args.use_cuda, split='Val')
        test_score, test_loss = eval_epoch(model, test_loader, criterion, args.use_cuda, split='Test')

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

    print()
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
