import torch
import matplotlib.pyplot as plt
import networkx as nx
import torch.nn.functional as F
from matplotlib.colors import ListedColormap

def get_attn_maps(data, attn_dict_list, attn_type='hard', return_onehot=False, rescale=False):
    """
    Args:
        # img: [B, C, H, W]
        data: PyG data object, including edge_index
        attn_dict_list: list[dict], each dict contains attention maps for a layer
        attn_type: str, type of attention map to extract (e.g., 'hard', 'soft')

    Returns:
        attn_maps: list[Tensor], attention map of shape [B, H, W, groups]
    """
    attn_maps = []
    with torch.no_grad():
        prev_attn_masks = None
        for idx, attn_dict in enumerate(attn_dict_list):
            if attn_dict is None:
                assert idx == len(attn_dict_list) - 1, 'only last layer can be None'
                continue
            # [B, G, HxW]
            # B: batch size (1), nH: number of heads, G: number of group token
            # get soft or hard attention maps
            attn_masks = attn_dict[attn_type]
            # [B, nH, G, HxW] -> [B, nH, HxW, G]
            # attn_masks = rearrange(attn_masks, 'b h g n -> b h n g')
            attn_masks = attn_masks.permute(0, 1, 3, 2)
            if prev_attn_masks is None:
                prev_attn_masks = attn_masks
            else:
                prev_attn_masks = prev_attn_masks @ attn_masks
            # [B, nH, HxW, G] -> [B, nH, H, W, G]
            # attn_maps.append(resize_attn_map(prev_attn_masks, *img.shape[-2:]))
            # attn_maps.append = prev_attn_masks[:, :, :data[i].num_nodes]
            # print("prev_attn_masks.shape = ", prev_attn_masks.shape)
            attn_maps.append(prev_attn_masks)
            # print("attn_maps.shape = ", prev_attn_masks.shape)


    for i in range(len(attn_maps)):
        attn_map = attn_maps[i]
        # [B, nh, H, W, G]
        assert attn_map.shape[1] == 1
        # [B, H, W, G]
        attn_map = attn_map.squeeze(1)

        if rescale:
            pass
            # attn_map = rearrange(attn_map, 'b h w g -> b g h w')
            # attn_map = F.interpolate(
            #     attn_map, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)
            # attn_map = rearrange(attn_map, 'b g h w -> b h w g')

        if return_onehot:
            # [B, H, W, G]
            attn_map = F.one_hot(attn_map.argmax(dim=-1), num_classes=attn_map.shape[-1]).to(dtype=attn_map.dtype)

        attn_maps[i] = attn_map

    return attn_maps


def plot_batch_graphs_all_layers(data, attn_dict_list, attn_type='hard', group_colors=None, title="Node Class Visualization (from hard_list)"):
    """
    Plot multiple graphs with node colors based on attn_dict, the class of each node.
    
    Parameters:
    - data: PyG data object, including edge_index
    - attn_dict_list: List of Tensors, each of shape [batch_size, num_nodes, num_classes] or [num_nodes, num_classes]
    - attn_type: Type of attention map to extract (e.g., 'hard', 'soft')
    - group_colors: optional list of colors per class
    - title: Title for the plot
    """
    
    batch_size = attn_dict_list[0].get(attn_type, 0).shape[0]
    attn_maps = get_attn_maps(data, attn_dict_list, attn_type=attn_type, return_onehot=False, rescale=False)
    # print("attn_maps = ", attn_maps)

    for i in range(batch_size):
        for idx, attn_map in enumerate(attn_maps):
            if attn_map is None:
                assert idx == len(attn_maps) - 1, 'only last layer can be None'
                continue
            # print("index = ", idx)
            # print("attn_map shape = ", attn_map.shape)
            # print("attn_map = ", attn_map)
            attn_i = attn_map[i] if attn_map is not None else None
            # print("attn_i shape = ", attn_i.shape)
            attn_i_transposed = attn_i.permute(1, 0)  # [num_classes, num_nodes]
            attn_i_trimmed = attn_i_transposed[:, :data[i].num_nodes] if attn_i is not None else None
            plot_single_graph(data=data[i], attn_dict=attn_i_trimmed, title=f"Node_Visualization_Graph_{i}_{attn_type}_Layer_{idx}")


    # for i in range(batch_size):
    #     hard_list = [attn_dict.get("hard") for attn_dict in attn_dict_list]
    #     hard_i = [hard[i] if hard is not None else None for hard in hard_list]
    #     hard_i_trimmed = [h[:, :, :data[i].num_nodes] if h is not None else None for h in hard_i]
    #     plot_single_graph(data=data[i], attn_dict=hard_i_trimmed, title=f"Node_Visualization_Graph_{i}")


def plot_batch_graphs(data, attn_dict_list, attn_type='hard', group_colors=None, title="Node Class Visualization (from hard_list)"):
    """
    Plot multiple graphs with node colors based on attn_dict, the class of each node.
    
    Parameters:
    - data: PyG data object, including edge_index
    - attn_dict_list: Tensor of shape [batch_size, num_nodes, num_classes] or [num_nodes, num_classes]
    - group_colors: optional list of colors per class
    - title: Title for the plot
    """
    
    batch_size = attn_dict_list[0].get("hard", 0).shape[0]
    # for idx, attn_dict in enumerate(attn_dict_list):
    #     print("index = ", idx)
    #     print("attn_dict_list['soft'] = ", attn_dict["soft"])
    attn_type = attn_type.lower()

    for i in range(batch_size):
        hard_list = attn_dict_list[0].get(attn_type)
        # hard_list = attn_dict_list[0].get("soft")
        hard_i = hard_list[i] if hard_list is not None else None
        hard_i_trimmed = hard_i[:, :, :data[i].num_nodes]
        plot_single_graph(data=data[i], attn_dict=hard_i_trimmed, title=f"Node_Visualization_Graph_{i}_{attn_type}")


def get_group_colors(num_classes):
    """
    Generate a list of colors for each class.
    
    Parameters:
    - num_classes: Number of classes
    
    Returns:
    - group_colors: List of colors for each class
    """
    cmap1 = plt.get_cmap("tab20")
    cmap2 = plt.get_cmap("tab20b")
    cmap3 = plt.get_cmap("tab20c")
    cmap4 = plt.get_cmap("Set3")
    cmap = ListedColormap(cmap1.colors + cmap2.colors + cmap3.colors + cmap4.colors)
    return [cmap(i) for i in range(num_classes)]


def plot_single_graph(data, attn_dict, group_colors=None, title="Node Class Visualization (from hard_list)"):
    """
    plot a single graph with node colors based on attn_dict, the class of each node.
    
    Parameters:
    - data: PyG data object, including edge_index
    - attn_dict: Tensor of shape [1, num_nodes, num_classes] or [num_nodes, num_classes]
    - group_colors: optional list of colors per class
    - title: Title for the plot
    """

    # print("hard tensor = ", attn_dict)
    # print("attn_dict.shape:", attn_dict.shape)

    # print("attn_dict.dim() = ", attn_dict.dim())
    # attn_dict = attn_dict.squeeze(1) if attn_dict.dim() == 4 else attn_dict
    # print("attn_dict.dim() = ", attn_dict.dim())
    if attn_dict.dim() == 3:
        attn_dict = attn_dict.squeeze(0)  # Remove batch dim => [num_nodes, num_classes]

    class_idx = attn_dict.argmax(dim=0).cpu().numpy()  # [num_nodes]
    # print("Class indices:", class_idx)
    # print("class index type:", type(class_idx))

    # print("attn_dict size:", attn_dict.size())
    num_classes = attn_dict.size(0)
    # print("Number of classes:", num_classes)
    if group_colors is None:
        # cmap = plt.get_cmap("tab10")
        # group_colors = [cmap(i) for i in range(num_classes)]
        group_colors = get_group_colors(num_classes)
    # print("Group colors:", group_colors)

    G = nx.Graph()
    G.add_edges_from(data.edge_index.cpu().t().tolist())

    node_colors = [group_colors[c] for c in class_idx]
    # print("Node colors:", node_colors)


    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, nodelist=sorted(G.nodes()), node_color=node_colors, node_size=1000)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels={i: f'{i}\nC{c}' for i, c in enumerate(class_idx)}, font_color='white')
    # for i, c in enumerate(class_idx):
    #     print(f"Node {i}: Class {c}, Color: {group_colors[c]}")

    for i, color in enumerate(group_colors):
        plt.scatter([], [], color=color, label=f'Class {i}')
    plt.legend(loc='lower left')
    plt.title(title)
    plt.axis('off')
    # plt.show()
    plt.savefig(f'./plots/{title}.png', bbox_inches='tight')
