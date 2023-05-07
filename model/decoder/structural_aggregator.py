import torch
import torch as th
import torch.nn as nn
from dgl.udf import EdgeBatch, NodeBatch
import dgl


class TreeAggregatorCell(nn.Module):
    def __init__(self, x_size: int, h_size: int, edge_time: float):
        super(TreeAggregatorCell, self).__init__()
        self.edge_time = edge_time
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size, bias=False)
        self.W_f = nn.Linear(x_size, h_size, bias=False)
        self.b_f = nn.Parameter(th.zeros(1, h_size))

    def message_func(self, edges: EdgeBatch):
        if self.edge_time:
            return {'h': edges.src['h'], 'c': edges.src['c'], 'time': edges.data['time']}
        else:
            return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes: NodeBatch):
        # [N,num_sum,hidden_dim]
        if self.edge_time:
            nodes.mailbox['h'] += nodes.mailbox['time']
        f = th.sigmoid(self.U_f(nodes.mailbox['h']) + self.W_f(nodes.data['x']).unsqueeze(dim=1) + self.b_f)
        c = th.sum(f * nodes.mailbox['c'], 1)
        # [N,hidden_dim]
        h_tild = th.sum(nodes.mailbox['h'], 1)
        return {'h_tild': h_tild, 'c': c}

    def apply_node_func(self, nodes: NodeBatch):
        iou = self.U_iou(nodes.data['h_tild']) + self.W_iou(nodes.data['x']) + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data['c']
        h = o * th.tanh(c)
        return {'h': h, 'c': c}


class TreeAggregator(nn.Module):
    def __init__(self, x_size: int, h_size: int, device: torch.device = None, dropout: float = 0.1,
                 edge_time: bool = False):
        super(TreeAggregator, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.device = device
        self.dropout = nn.Dropout(p=dropout)
        cell = TreeAggregatorCell
        self.cell = cell(x_size, h_size, edge_time)

    def forward(self, g: dgl.DGLHeteroGraph) -> torch.Tensor:
        """
        Aggregate the user embeddings along the graph structural
        :param g: a dgl graph
        :return: the embeddings of cascades
        """
        # feed embedding
        g.ndata['h'] = torch.zeros(g.num_nodes(), self.h_size).to(self.device)
        g.ndata['c'] = torch.zeros(g.num_nodes(), self.h_size).to(self.device)
        g.ndata['h_tild'] = torch.zeros(g.num_nodes(), self.h_size).to(self.device)
        # propagate
        dgl.prop_nodes_topo(g, self.cell.message_func, self.cell.reduce_func, apply_node_func=self.cell.apply_node_func)
        h = dgl.readout_nodes(g, 'h', 'mask') / dgl.readout_nodes(g, 'mask')
        return self.dropout(h)
