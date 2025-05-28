# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch_geometric.nn as gnn
from torch.nn.utils.rnn import pad_sequence
from .layers import TransformerEncoderGroupingLayer
from groupvit.groupvit import GroupingBlock, MixerMlp, GroupingLayer
from einops import repeat
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from model.gin import Encoder
from experiments.arguments import load_args

class TwoLayerGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TwoLayerGCNConv, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x

class GraphGroupingLayer(nn.Module):
    """GroupingLayer in SAT."""
    def __init__(self,
                 dim,
                 num_input_token,
                 depth,
                 num_heads,
                 num_group_token,
                 edge_index,
                 complete_edge_index,
                 subgraph_node_index=None,
                 subgraph_edge_index=None,
                 subgraph_edge_attr=None,
                 subgraph_indicator_index=None,
                 edge_attr=None,
                 degree=None,
                 ptr=None,
                 downsample=None,
                 use_checkpoint=False,
                 group_projector=None,
                 zero_init_group_token=False,
                 gnn_type="graph",
                 **kwargs):
        super().__init__()
        
        self.dim = dim
        self.num_input_token = num_input_token
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.num_group_token = num_group_token
        self.downsample = downsample
        self.group_projector = group_projector
        self.kwargs = kwargs

        # Initialize group token if needed
        if num_group_token > 0:
            self.group_token = nn.Parameter(torch.zeros(1, num_group_token, dim))
            # self.group_token = nn.Parameter(torch.zeros(1, num_group_token, dim))
            if not zero_init_group_token:
                nn.init.trunc_normal_(self.group_token, std=.02)
        else:
            self.group_token = None

        # Build transformer blocks (layers)
        blocks = []
        for _ in range(depth):
            blocks.append(
                TransformerEncoderGroupingLayer(
                    d_model=dim,
                    edge_index=edge_index,
                    complete_edge_index=complete_edge_index,
                    subgraph_node_index=subgraph_node_index,
                    subgraph_edge_index=subgraph_edge_index,
                    subgraph_edge_attr=subgraph_edge_attr,
                    subgraph_indicator_index=subgraph_indicator_index,
                    edge_attr=edge_attr,
                    degree=degree,
                    ptr=ptr,
                    nhead=num_heads,
                    gnn_type=gnn_type,
                    **self.kwargs
                ).to(torch.device("cuda"))
            )
        self.blocks = nn.ModuleList(blocks)

    @property
    def with_group_token(self):
        return self.group_token is not None

    def split_x(self, x):
        if self.with_group_token:
            return x[:, :-self.num_group_token], x[:, -self.num_group_token:]
        else:
            return x, None

    def concat_x(self, x, group_token=None):
        if group_token is None:
            return x
        group_token = group_token.to(x.device)
        # print("group_token.shape = ", group_token.shape)
        return torch.cat([x, group_token], dim=1)

    def forward(self, x, prev_group_token=None, return_attn=False):
        """
        Args:
            x (torch.Tensor): image token, [B, L, C]
            prev_group_token (torch.Tensor): group token, [B, S_1, C]
            return_attn (bool): whether to return attention maps
        """
        if self.with_group_token:
            group_token = self.group_token.expand(x.size(0), -1, -1)
            # group_token = self.group_token.expand(x.size(0), -1)
            # group_token = self.group_token.expand(x.size(0), -1, -1)
            if self.group_projector is not None:
                # group_token = group_token.to(prev_group_token.device)
                # print("group_token device = ", group_token.device)
                prev_group_token = prev_group_token.to(group_token.device)
                group_token = group_token + self.group_projector(prev_group_token)
        else:
            group_token = None

        if(len(x.shape) == 2):
            x = x.unsqueeze(2)
            x = x.expand(-1, -1, self.dim)

        # B, L, C = x.shape
        cat_x = self.concat_x(x, group_token)
        # print("cat_x.shape = ", cat_x.shape)
        for blk_idx, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                cat_x = checkpoint.checkpoint(blk, cat_x)
            else:
                cat_x = blk(cat_x)

        x, group_token = self.split_x(cat_x)

        attn_dict = None
        if self.downsample is not None:
            x, attn_dict = self.downsample(x, group_token, return_attn=return_attn)

        return x, group_token, attn_dict


class GroupGraphTransformer(nn.Module):
    def __init__(self, in_size, num_class, d_model, num_heads=8,
                 dim_feedforward=512, dropout=0.0, num_layers=4,
                 batch_norm=False, abs_pe=False, abs_pe_dim=0,
                 gnn_type="graph", se="gnn", use_edge_attr=False, num_edge_features=4,
                 in_embed=True, edge_embed=True, use_global_pool=True, max_seq_len=None,
                 global_pool='mean', depths=[6, 3, 3], **kwargs):
        super().__init__()
        # self.num_layers = num_layers
        print("initializing GraphTransformer")
        self.num_layers = len(depths)
        self.depths = depths

        self.abs_pe = abs_pe
        self.abs_pe_dim = abs_pe_dim
        if abs_pe and abs_pe_dim > 0:
            self.embedding_abs_pe = nn.Linear(abs_pe_dim, d_model)
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
        
        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            edge_dim = kwargs.get('edge_dim', 32)
            if edge_embed:
                if isinstance(num_edge_features, int):
                    self.embedding_edge = nn.Embedding(num_edge_features, edge_dim)
                else:
                    raise ValueError("Not implemented!")
            else:
                self.embedding_edge = nn.Linear(in_features=num_edge_features,
                    out_features=edge_dim, bias=False)
        else:
            kwargs['edge_dim'] = None

        self.gnn_type = gnn_type
        self.se = se
        self.kwargs = kwargs
        self.global_pool = global_pool
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        elif global_pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, d_model))
            self.pooling = None
        self.use_global_pool = use_global_pool

        self.max_seq_len = max_seq_len
        if max_seq_len is None:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(True),
                nn.Linear(d_model, num_class)
            )
        else:
            self.classifier = nn.ModuleList()
            for i in range(max_seq_len):
                self.classifier.append(nn.Linear(d_model, num_class))
        print("init complete")

    def build_encoder(self, dim, num_input_token,
                        edge_index, complete_edge_index,
                        subgraph_node_index=None, subgraph_edge_index=None,
                        subgraph_edge_attr=None, subgraph_indicator_index=None,
                        edge_attr=None, degree=None, ptr=None,
                        downsample=None, use_checkpoint=False,
                        group_projector=None, zero_init_group_token=False, 
                        num_classes=0, embed_dim=384, embed_factors=[1, 1, 1],
                        depths=[6, 3, 3], num_heads=[8, 8, 8], num_group_tokens=[64, 8, 0], 
                        num_output_groups=[64, 8], hard_assignment=True):

        self.layers = nn.ModuleList()
        norm_layer = nn.LayerNorm
        self.num_group_tokens = num_group_tokens

        for i_layer in range(self.num_layers):

            dim = int(dim * embed_factors[i_layer])
            # dim = int(embed_dim * embed_factors[i_layer])
            downsample = None
            if i_layer < self.num_layers - 1:
                out_dim = dim * embed_factors[i_layer + 1]
                # out_dim = embed_dim * embed_factors[i_layer + 1]
                downsample = GroupingBlock(
                    dim=dim,
                    out_dim=out_dim,
                    num_heads=num_heads[i_layer],
                    num_group_token=num_group_tokens[i_layer],
                    num_output_group=num_output_groups[i_layer],
                    norm_layer=norm_layer,
                    hard=hard_assignment,
                    gumbel=hard_assignment).to(torch.device("cuda"))
                num_output_token = num_output_groups[i_layer]

            if i_layer > 0 and num_group_tokens[i_layer] > 0:
                prev_dim = int(dim * embed_factors[i_layer - 1])
                # prev_dim = int(embed_dim * embed_factors[i_layer - 1])
                group_projector = nn.Sequential(
                    norm_layer(prev_dim),
                    MixerMlp(num_group_tokens[i_layer - 1], prev_dim // 2, num_group_tokens[i_layer]))

                if dim != prev_dim:
                    group_projector = nn.Sequential(group_projector, norm_layer(prev_dim),
                                                    nn.Linear(prev_dim, dim, bias=False))
            else:
                group_projector = None
            layer = GraphGroupingLayer(
                dim=dim,
                num_input_token=num_input_token,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                num_group_token=num_group_tokens[i_layer],
                edge_index=edge_index,
                complete_edge_index=complete_edge_index,
                subgraph_node_index=subgraph_node_index,
                subgraph_edge_index=subgraph_edge_index,
                subgraph_edge_attr=subgraph_edge_attr,
                subgraph_indicator_index=subgraph_indicator_index,
                edge_attr=edge_attr,
                degree=degree,
                ptr=ptr,
                downsample=downsample,
                use_checkpoint=use_checkpoint,
                group_projector=group_projector,
                zero_init_group_token=zero_init_group_token,
                gnn_type=self.gnn_type,
                **self.kwargs
            )
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                num_input_token = num_output_token


    def forward(self, data, return_attn=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        node_depth = data.node_depth if hasattr(data, "node_depth") else None
        
        if self.se == "khopgnn":
            subgraph_node_index = data.subgraph_node_idx
            subgraph_edge_index = data.subgraph_edge_index
            subgraph_indicator_index = data.subgraph_indicator 
            subgraph_edge_attr = data.subgraph_edge_attr if hasattr(data, "subgraph_edge_attr") \
                                    else None
        else:
            subgraph_node_index = None
            subgraph_edge_index = None
            subgraph_indicator_index = None
            subgraph_edge_attr = None

        complete_edge_index = data.complete_edge_index if hasattr(data, 'complete_edge_index') else None
        abs_pe = data.abs_pe if hasattr(data, 'abs_pe') else None
        degree = data.degree if hasattr(data, 'degree') else None
        output = self.embedding(x) if node_depth is None else self.embedding(x, node_depth.view(-1,))
            
        if self.abs_pe and abs_pe is not None:
            abs_pe = self.embedding_abs_pe(abs_pe)
            output = output + abs_pe
        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.embedding_edge(edge_attr)
            if subgraph_edge_attr is not None:
                subgraph_edge_attr = self.embedding_edge(subgraph_edge_attr)
        else:
            edge_attr = None
            subgraph_edge_attr = None

        if self.global_pool == 'cls' and self.use_global_pool:
            bsz = len(data.ptr) - 1
            if complete_edge_index is not None:
                new_index = torch.vstack((torch.arange(data.num_nodes).to(data.batch), data.batch + data.num_nodes))
                new_index2 = torch.vstack((new_index[1], new_index[0]))
                idx_tmp = torch.arange(data.num_nodes, data.num_nodes + bsz).to(data.batch)
                new_index3 = torch.vstack((idx_tmp, idx_tmp))
                complete_edge_index = torch.cat((
                    complete_edge_index, new_index, new_index2, new_index3), dim=-1)
            if subgraph_node_index is not None:
                idx_tmp = torch.arange(data.num_nodes, data.num_nodes + bsz).to(data.batch)
                subgraph_node_index = torch.hstack((subgraph_node_index, idx_tmp))
                subgraph_indicator_index = torch.hstack((subgraph_indicator_index, idx_tmp))
            degree = None
            cls_tokens = repeat(self.cls_token, '() d -> b d', b=bsz)
            output = torch.cat((output, cls_tokens))
        # print("dim = ", self.embedding.embedding_dim)
        self.build_encoder(
            dim=self.embedding.embedding_dim,
            # embed_dim=x.size(-1),
            num_input_token=output.size(1),
            edge_index=edge_index,
            complete_edge_index=complete_edge_index,
            subgraph_node_index=subgraph_node_index,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_edge_attr=subgraph_edge_attr,
            subgraph_indicator_index=subgraph_indicator_index,
            edge_attr=edge_attr,
            degree=degree,
            depths=self.depths,
            ptr=data.ptr if hasattr(data, 'ptr') else None,
        )
        i_layer = 0
        for layer in self.layers:
            if i_layer == 0:
                x, group_token, attn_dicts = layer(
                    output,
                    prev_group_token=None,
                    return_attn=return_attn
                )
            else:
                x, group_token, attn_dict = layer(
                    x,
                    prev_group_token=group_token,
                    return_attn=return_attn
                )
            i_layer += 1

        output = x
        output = torch.mean(output, dim=2) if output.dim() == 3 else output

        # readout step
        if self.use_global_pool:
            if self.global_pool == 'cls':
                output = output[-bsz:]
            else:
                output = self.pooling(output, data.batch)
        if self.max_seq_len is not None:
            pred_list = []
            for i in range(self.max_seq_len):
                pred_list.append(self.classifier[i](output))
            return pred_list
        
        linear_out = nn.Linear(output.size(1), self.embedding.embedding_dim).to(output.device)
        output = linear_out(output)
        return self.classifier(output)


class GraphViT(nn.Module):
    def __init__(self, in_size, d_model, num_class, abs_pe=False, abs_pe_dim=0, in_embed=True, subgraph_embed=False,
                 embed_factors=[1, 1, 1], depths=[6, 3, 3],
                 num_heads=[8, 8, 8], num_group_tokens=[64, 8, 0],
                 num_output_groups=[64, 8], hard_assignment=True,
                 mlp_ratio=4, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., 
                 drop_path_rate=0.1, use_ckpt=False, global_pool='mean', max_seq_len=None, **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_factors = embed_factors
        self.depths = depths
        self.num_group_tokens = num_group_tokens
        self.num_output_groups = num_output_groups
        self.num_heads = num_heads
        self.hard_assignment = hard_assignment
        # GroupingLayer parameters
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.use_checkpoint = use_ckpt
        
        # get positional embeddings
        self.abs_pe = abs_pe
        self.abs_pe_dim = abs_pe_dim
        if abs_pe and abs_pe_dim > 0:
            self.embedding_abs_pe = nn.Linear(abs_pe_dim, d_model)

        # get input embeddings
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
        self.subgraph_embed = subgraph_embed
        # Aggregrate
        self.args = load_args()
        
        if self.args.use_gcn:
            self.aggregate_func = TwoLayerGCNConv(d_model, d_model)
        # load pre-trained GIN model
        elif self.args.use_pretrained_gin:
            checkpoint = torch.load(f'ckpt/{self.args.DS}/best_model_0.2_0.pth')

            # 取出 encoder 部分的 state_dict，去掉 'encoder.' prefix
            encoder_state_dict = {k[len('encoder.'):]: v for k, v in checkpoint.items() if k.startswith('encoder.')}
            filtered_state_dict = {k: v for k, v in encoder_state_dict.items() if not (k.startswith('convs.0.') or k.startswith('bns.0.'))}
            # print("filtered_state_dict keys:", filtered_state_dict.keys())
            self.aggregate_func = Encoder(num_features=d_model, dim=d_model, num_gc_layers=5)

            # 載入到 new_model，strict=False 避免因缺少第一層參數報錯
            self.aggregate_func.load_state_dict(filtered_state_dict, strict=False)
            self.aggregate_func = self.aggregate_func.to(torch.device("cuda"))
            self.aggregate_func.eval()
        else:
            self.aggregate_func = None

        # pooling
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        elif global_pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, d_model))
            self.pooling = None

        # classifier
        self.max_seq_len = max_seq_len
        if max_seq_len is None:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(True),
                nn.Linear(d_model, num_class)
            )
        else:
            self.classifier = nn.ModuleList()
            for i in range(max_seq_len):
                self.classifier.append(nn.Linear(d_model, num_class))
            
    def build_layers(self, d_model, num_input_token):
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]

        # build grouping layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dim = int(d_model * self.embed_factors[i_layer])
            # downsample
            downsample = None
            if i_layer < self.num_layers - 1:
                out_dim = int(d_model * self.embed_factors[i_layer + 1])
                downsample = GroupingBlock(
                    dim=dim,
                    out_dim=out_dim,
                    num_heads=self.num_heads[i_layer],
                    num_group_token=self.num_group_tokens[i_layer],
                    num_output_group=self.num_output_groups[i_layer],
                    norm_layer=nn.LayerNorm,
                    hard=self.hard_assignment,
                    gumbel=self.hard_assignment
                )
                num_output_token = self.num_output_groups[i_layer]

            # group projector
            if i_layer > 0 and self.num_group_tokens[i_layer] > 0:
                prev_dim = int(d_model * self.embed_factors[i_layer - 1])
                group_projector = nn.Sequential(
                    nn.LayerNorm(prev_dim),
                    MixerMlp(self.num_group_tokens[i_layer - 1], prev_dim // 2, self.num_group_tokens[i_layer]))

                if dim != prev_dim:
                    group_projector = nn.Sequential(group_projector, nn.LayerNorm(prev_dim),
                                                    nn.Linear(prev_dim, dim, bias=False))
            else:
                group_projector = None

            layer = GroupingLayer(
                dim=dim,
                num_input_token=num_input_token,
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                num_group_token=self.num_group_tokens[i_layer],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                downsample=downsample,
                use_checkpoint=self.use_checkpoint,
                group_projector=group_projector,
                # only zero init group token if we have a projection
                zero_init_group_token=group_projector is not None)
            self.layers.append(layer.cuda())



    def split_edge_index(self, edge_index, ptr):
        # 假設 edge_index: [2, E]
        # 假設 data.ptr: [B+1]，節點起訖位置，例如：[0, N1, N1+N2, ..., total_nodes]
        # ptr: LongTensor shape [B+1]
        B = ptr.size(0) - 1
        E = edge_index.size(1)

        # 取得每條邊兩端節點的 batch id（子圖 id）
        # 先做搜尋排序定位節點在哪個ptr區間內
        # ptr 是遞增的，用 searchsorted 找對應 batch_id

        src_batch = torch.searchsorted(ptr, edge_index[0], right=True) - 1  # shape: [E]
        dst_batch = torch.searchsorted(ptr, edge_index[1], right=True) - 1  # shape: [E]

        # 篩選兩端都屬於同一子圖的邊
        mask = (src_batch == dst_batch)

        # 篩選出來的邊與對應 batch id
        filtered_batch = src_batch[mask]
        filtered_edge_index = edge_index[:, mask]

        # 重定位邊的節點索引：都減去該子圖起始節點 id
        node_offset = ptr[filtered_batch]  # shape: [num_filtered_edges]
        filtered_edge_index = filtered_edge_index - node_offset.unsqueeze(0)

        # 接下來把每個子圖的邊分組

        # 找出每個 batch 裡邊數量
        edge_counts = torch.bincount(filtered_batch, minlength=B)  # shape: [B]

        # 使用 torch.split 依照每張圖邊數切割
        edge_splits = torch.split(filtered_edge_index, edge_counts.tolist(), dim=1)

        return edge_splits


    def forward(self, data, return_attn=False):
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # print("x shape:", x.shape, "x dtype:", x.dtype)
        # print("x = ", x[0])
        if self.subgraph_embed:
            pass
        else:
            node_depth = data.node_depth if hasattr(data, "node_depth") else None

            # positional + input embeddings
            abs_pe = data.abs_pe if hasattr(data, 'abs_pe') else None
            # print("abs_pe = ", abs_pe.shape if abs_pe is not None else None)
            output = self.embedding(x) if node_depth is None else self.embedding(x, node_depth.view(-1,))
            # output = self.gcn_conv(output, edge_index)
            if self.abs_pe and abs_pe is not None:
                abs_pe = self.embedding_abs_pe(abs_pe)
                output = output + abs_pe

            # GCNConv
            # output = self.gcn_conv(output, edge_index)
            # output = self.encoder(output, edge_index, batch)
            # print("output.type = ", type(output))

            split_sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()  # [num_nodes_graph1, num_nodes_graph2, ...]
            node_embeddings = output.split(split_sizes)  # list of [Ni, dim] tensors, i=1..B

            if self.aggregate_func is not None:
                node_embeddings = list(node_embeddings)
                edge_split = self.split_edge_index(edge_index, data.ptr)  # list of [2, Ni] tensors, i=1..B
                for i in range(len(node_embeddings)):
                    # print("self.aggregate_func = ", self.aggregate_func)
                    node_embeddings[i] = self.aggregate_func(node_embeddings[i], edge_split[i].to(node_embeddings[i].device))

            # pad into [batch_size, max_nodes, dim]
            output_padded = pad_sequence(node_embeddings, batch_first=True)  # [B, N_max, dim]
            B, N_max, dim = output_padded.shape

        # build layers
        self.build_layers(
            d_model=dim,
            num_input_token=N_max
        )

        # forward layers
        group_token = None
        attn_dict_list = []
        for layer in self.layers:
            output_padded, group_token, attn_dict = layer(
                output_padded,
                prev_group_token=group_token,
                return_attn=return_attn
            )
            attn_dict_list.append(attn_dict)

        # readout step
        device = output_padded.device
        mask = torch.zeros((len(split_sizes), output_padded.size(1)), dtype=torch.bool, device=device)  # [B, N_max]
        for i, l in enumerate(split_sizes):
            mask[i, :l] = 1
        mask = mask.unsqueeze(-1)  # [B, N_max, 1]
        masked_output = output_padded * mask  # [B, N_max, C], padding 會是 0
        pooled = masked_output.sum(dim=1) / mask.sum(dim=1)  # [B, C]

        # output = self.pooling(output_padded, data.batch) if self.pooling is not None else output_padded

        # print("output.shape = ", output.shape) #  = [2960, 64]
        # print("output = ", output[0])
        # print("output_padded.shape = ", output_padded.shape)
        # print("output_padded = ", output_padded[0])

        if self.max_seq_len is not None:
            pred_list = []
            for i in range(self.max_seq_len):
                pred_list.append(self.classifier[i](pooled))
            return pred_list

        # graph regression
        # linear_out = nn.Linear(pooled.size(1), self.embedding.embedding_dim).to(pooled.device)
        # graph classification
        linear_out = nn.Linear(pooled.size(1), self.embedding.out_features).to(pooled.device)
        pooled = linear_out(pooled)
        return self.classifier(pooled)