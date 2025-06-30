import os
import argparse
import torch
from sat.position_encoding import POSENCODINGS
from sat.gnn_layers import GNN_TYPES

def load_args():
    parser = argparse.ArgumentParser(
        description='Structure-Aware Transformer on ZINC',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--dataset', type=str, default="ZINC",
                        help='name of dataset')
    parser.add_argument('--num-heads', type=int, default=8, help="number of heads")
    parser.add_argument('--num-layers', type=int, default=6, help="number of layers")
    parser.add_argument('--dim-hidden', type=int, default=64, help="hidden dimension of Transformer")
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout")
    parser.add_argument('--epochs', type=int, default=2000,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--abs-pe', type=str, default=None, choices=POSENCODINGS.keys(),
                        help='which absolute PE to use?')
    parser.add_argument('--abs-pe-dim', type=int, default=20, help='dimension for absolute PE')
    parser.add_argument('--outdir', type=str, default='',
                        help='output path')
    parser.add_argument('--warmup', type=int, default=5000, help="number of iterations for warmup")
    parser.add_argument('--layer-norm', action='store_true', help='use layer norm instead of batch norm')
    parser.add_argument('--use-edge-attr', action='store_true', help='use edge features')
    parser.add_argument('--edge-dim', type=int, default=32, help='edge features hidden dim')
    parser.add_argument('--gnn-type', type=str, default='graphsage',
                        choices=GNN_TYPES,
                        help="GNN structure extractor type")
    parser.add_argument('--k-hop', type=int, default=2, 
        help="Number of hops to use when extracting subgraphs around each node")
    parser.add_argument('--global-pool', type=str, default='mean', choices=['mean', 'cls', 'add'],
                        help='global pooling method')
    parser.add_argument('--se', type=str, default="gnn", 
            help='Extractor type: khopgnn, or gnn')
    
    # add:
    parser.add_argument('--unsupervised', action='store_true', help='supervised or unsupervised', default=False)
    parser.add_argument('--model', type=str, default='sat', choices=['sat', 'groupvit', 'graphvit'], help='model type')
    parser.add_argument('--DS', type=str, default='MUTAG', choices=['MUTAG', 'PROTEINS', 'DD', 'NCI1', 'IMDB-B', 'COLLAB', 'ENZYMES', 'MSRC-21'], help='dataset for graph classification')
    parser.add_argument('--data_path', type=str, default='/home/qiannnhui/data/data', help='path to dataset') # 249
    parser.add_argument('--subgraph_embed', action='store_true', help='use subgraph embedding', default=False)
    parser.add_argument('--lr_schedule', type=str, default='none', choices=['cosine', 'step', 'none'], help='learning rate schedule')

    # aggregrate
    parser.add_argument('--use_gcn', action='store_true', help='aggregate with 2 layer gcn', default=False)
    parser.add_argument('--use_pretrained_gin', action='store_true', help='aggregrate with pretrained gin', default=False)

    # OGB dataset
    parser.add_argument('--not_extract_node_feature', action='store_true')
    parser.add_argument('--aggr', type=str, default='add',
                        help='the aggregation operator to obtain nodes\' initial features [mean, max, add]')
    parser.add_argument('--plot_attn', action='store_true', help='plot attention weights')
    parser.add_argument('--graph_idx', type=int, default=-1,
                        help='index of the graph to visualize attention weights')
    
    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    args.batch_norm = not args.layer_norm

    args.save_logs = False
    if args.outdir != '':
        args.save_logs = True
        outdir = args.outdir
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/{}'.format(args.dataset)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/seed{}'.format(args.seed)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        if args.use_edge_attr:
            outdir = outdir + '/edge_attr'
            if not os.path.exists(outdir):
                try:
                    os.makedirs(outdir)
                except Exception:
                    pass
        pedir = 'None' if args.abs_pe is None else '{}_{}'.format(args.abs_pe, args.abs_pe_dim)
        outdir = outdir + '/{}'.format(pedir)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        bn = 'BN' if args.batch_norm else 'LN'
        if args.se == "khopgnn":
            outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.se, args.gnn_type, args.k_hop, args.dropout, args.lr, args.weight_decay,
                args.num_layers, args.num_heads, args.dim_hidden, bn,
            )

        else:
            outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.gnn_type, args.k_hop, args.dropout, args.lr, args.weight_decay,
                args.num_layers, args.num_heads, args.dim_hidden, bn,
            )
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        args.outdir = outdir
    return args