def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import torch
import torch.nn as nn
from model.gin import Encoder
from model import *

class simclr(nn.Module):
  def __init__(self, d_model, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
    super(simclr, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    in_size = 1

    self.embedding_dim = mi_units = d_model * num_gc_layers
    self.encoder = Encoder(in_size, d_model, num_gc_layers)
    self.embedding = nn.Embedding(in_size, d_model)

    in_embed = True
    if in_embed:
        if isinstance(in_size, int):
            self.embedding = nn.Embedding(in_size, d_model) 
        elif isinstance(in_size, nn.Module):
            self.embedding = in_size
        else:
            raise ValueError("Not implemented!")
    else:
        self.embedding = nn.Linear(in_features=in_size,
                                    out_features=d_model,
                                    bias=False)
            


    self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))

    self.init_emb()

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


  def forward(self, data):
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    batch = data.batch
    node_depth = data.node_depth if hasattr(data, "node_depth") else None
    # print("x shape:", x.shape)  # 確保 x 的形狀是 [num_nodes, num_features]
    # print("edge_index shape:", edge_index.shape)  # 確保 edge_index 的形狀是 [2, num_edges]

    if x is None:
        x = torch.ones(batch.shape[0]).to(device='cuda')
    # x = self.embedding(x) if self.embedding is not None else self.embedding(x, node_depth.view(-1, 1))
    if x.dim() == 1:
        x = x.view(-1, 1)
    # print("x shape after Linear:", x.shape)  # 確保 x 的形狀是 [num_nodes, num_features]

    y, M = self.encoder(x, edge_index, batch)
    
    y = self.proj_head(y)

    # max pooling
    y = y.sum(dim=1, keepdim=True)
    # x = x.mean(dim=1, keepdim=True)  
    # x, _ = x.max(dim=1, keepdim=True)  # 沿著第二維度進行最大值池化，保留維度
    # y = y.expand(-1, 384)  # 或者 x1.repeat(1, 384)


    return y

  def infonce_loss(self, x, x_aug):

    T = 0.2
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()

    return loss