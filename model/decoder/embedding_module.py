import numpy as np
import torch
from torch import nn
from typing import Dict, Mapping, List, Union, Tuple
from model.encoder.state.dynamic_state import DynamicState
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from model.time_encoder import TimeSlotEncoder
from model.decoder.structural_aggregator import TreeAggregator
import torch.nn.modules.module
from utils.hgraph import HGraph
import dgl


class EmbeddingModule(nn.Module):
    def __init__(self, dynamic_state: Mapping[str, DynamicState], embedding_dimension: int, device: torch.device,
                 dropout: float, hgraph: HGraph):
        super(EmbeddingModule, self).__init__()
        self.dynamic_state = dynamic_state
        self.dropout = dropout
        self.embedding_dimension = embedding_dimension
        self.device = device
        self.hgraph = hgraph

    def compute_embedding(self, cascades: np.ndarray) -> \
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute the embedding of given cascades by different methods
        :param cascades: the ids of cascades, ndarray of shape (n_node)
        :return: the embedding of the corresponding cascades,tensor of shape (n_node,emb_dim)
        """
        ...


class IdentityEmbedding(EmbeddingModule):
    def compute_embedding(self, cascade):
        """return the dynamic states of cascades"""
        return self.dynamic_state['cas'].get_state(cascade, from_cache=True)


class ConcatEmbedding(EmbeddingModule):
    def __init__(self, dynamic_state: Mapping[str, DynamicState], embedding_dimension: int, device: torch.device,
                 dropout: float, hgraph: HGraph, max_global_time: float = 0.0, global_time_num: int = 0):
        super(ConcatEmbedding, self).__init__(dynamic_state, embedding_dimension, device,
                                              dropout, hgraph)
        self.time_embedding = TimeSlotEncoder(embedding_dimension, max_global_time, global_time_num)

    def compute_embedding(self, cascades):
        """concat the dynamic states of the sending user, receiving user and cascade in the last interaction"""
        cas_interaction = self.hgraph.batch_cas_info()
        cas_pub_times = torch.tensor(self.hgraph.get_cas_pub_time(cascades), dtype=torch.float, device=self.device)
        src_embs, dst_embs = [], []
        for cas in cascades:
            src, dst = zip(*cas_interaction[cas])
            src_emb = self.dynamic_state['user'].get_state(src, 'src', from_cache=True).squeeze(dim=0)
            dst_emb = self.dynamic_state['user'].get_state(dst, 'dst', from_cache=True).squeeze(dim=0)
            if len(src) > 1:
                src_emb = src_emb[-1]
                dst_emb = dst_emb[-1]
            src_embs.append(src_emb)
            dst_embs.append(dst_emb)
        cas_embs = self.dynamic_state['cas'].get_state(cascades, from_cache=True)
        cas_embs += self.time_embedding(cas_pub_times)
        src_embs = torch.stack(src_embs, dim=0)
        dst_embs = torch.stack(dst_embs, dim=0)
        return torch.cat([src_embs, dst_embs, cas_embs], dim=1)


class AggregateEmbedding(EmbeddingModule):
    def __init__(self, dynamic_state: Mapping[str, DynamicState], input_dimension: int, embedding_dimension: int,
                 device: torch.device, dropout: float, hgraph: HGraph, max_time: float, time_num: int,
                 use_static: bool, user_num: int, max_global_time: float, global_time_num: int, use_dynamic: bool,
                 use_temporal: bool, use_structural: bool):
        super(AggregateEmbedding, self).__init__(dynamic_state, embedding_dimension, device, dropout, hgraph)
        self.use_dynamic = use_dynamic
        self.use_temporal = use_temporal
        self.use_structural = use_structural
        self.use_static = use_static
        self.dropout = nn.Dropout(dropout)
        self.time_position_encoder = TimeSlotEncoder(embedding_dimension, max_time, time_num)
        self.position_embedding = nn.Embedding(100, embedding_dimension)
        if self.use_dynamic:
            dynamic_trans_input_dim = 3 * input_dimension
            self.concat_emb = ConcatEmbedding(dynamic_state, embedding_dimension, device, dropout, hgraph,
                                              max_global_time, global_time_num)
            if use_temporal:
                dynamic_trans_input_dim += input_dimension
                self.temporal_aggregator = nn.LSTM(input_size=input_dimension, hidden_size=embedding_dimension,
                                                   batch_first=True)
            if use_structural:
                dynamic_trans_input_dim += 2 * input_dimension
                self.structural_agg_root = TreeAggregator(input_dimension, embedding_dimension, device, edge_time=True)
                self.structural_agg_leaf = TreeAggregator(input_dimension, embedding_dimension, device, edge_time=True)
            self.trans = nn.Sequential(nn.Linear(in_features=dynamic_trans_input_dim, out_features=embedding_dimension),
                                       nn.ReLU())
        if self.use_static:
            static_trans_input_dim = 0
            self.static_state = nn.Embedding(num_embeddings=user_num, embedding_dim=embedding_dimension)
            nn.init.uniform_(self.static_state.weight, 0, 1)
            if use_temporal:
                self.static_rnn = nn.LSTM(input_size=input_dimension, hidden_size=embedding_dimension, batch_first=True)
                static_trans_input_dim += input_dimension
            if use_structural:
                self.static_root_gnn = TreeAggregator(input_dimension, embedding_dimension, device, edge_time=True)
                self.static_leaf_gnn = TreeAggregator(input_dimension, embedding_dimension, device, edge_time=True)
                static_trans_input_dim += 2 * input_dimension
            self.static_trans = nn.Sequential(
                nn.Linear(in_features=static_trans_input_dim, out_features=embedding_dimension),
                nn.ReLU())

    def compute_static_emb(self, cascades: np.ndarray, cas_history: torch.Tensor, length: List[int],
                           cas_time_emb: torch.Tensor, cas_pos_emb: torch.Tensor, graph_root: dgl.DGLHeteroGraph,
                           graph_leaf: dgl.DGLHeteroGraph) -> torch.Tensor:
        final_embedding = []
        if self.use_temporal:
            cas_his_emb = self.dropout(self.static_state(cas_history.reshape(-1)).reshape(*cas_history.shape, -1))
            cas_his_emb = cas_his_emb + cas_time_emb + cas_pos_emb
            cas_his_emb = pack_padded_sequence(cas_his_emb, lengths=length, batch_first=True, enforce_sorted=False)
            _, (h, c) = self.static_rnn.forward(cas_his_emb)
            final_embedding.append(h.squeeze(dim=0))
        if self.use_structural:
            root_feature = self.static_state(graph_root.ndata['id'])
            leaf_feature = self.static_state(graph_leaf.ndata['id'])
            graph_root.ndata['x'] = self.dropout(root_feature)
            graph_leaf.ndata['x'] = self.dropout(leaf_feature)
            root_emb, leaf_emb = self.static_root_gnn(graph_root), self.static_leaf_gnn(graph_leaf)
            final_embedding.extend([root_emb, leaf_emb])
        return self.static_trans(torch.cat(final_embedding, dim=1))

    def compute_dynamic_emb(self, cascades: np.ndarray, cas_history: torch.Tensor, length: List[int],
                            cas_time_emb: torch.Tensor, cas_pos_emb: torch.Tensor, graph_root: dgl.DGLHeteroGraph,
                            graph_leaf: dgl.DGLHeteroGraph) -> torch.Tensor:
        newest_dynamic_state = self.concat_emb.compute_embedding(cascades)
        final_embedding = [newest_dynamic_state]
        if self.use_temporal:
            cas_his_emb = self.dynamic_state['user'].get_state(cas_history.reshape(-1), 'src', from_cache=False). \
                reshape(*cas_history.shape, -1)
            cas_his_emb = cas_his_emb + cas_time_emb + cas_pos_emb
            cas_his_emb = pack_padded_sequence(cas_his_emb, lengths=length, batch_first=True, enforce_sorted=False)
            _, (h, c) = self.temporal_aggregator.forward(cas_his_emb)
            final_embedding.append(h.squeeze(dim=0))
        if self.use_structural:
            graph_root.ndata['x'] = self.dynamic_state['user'].get_state(graph_root.ndata['id'], 'dst',
                                                                         from_cache=False)
            graph_leaf.ndata['x'] = self.dynamic_state['user'].get_state(graph_leaf.ndata['id'], 'src',
                                                                         from_cache=False)
            root_emb, leaf_emb = self.structural_agg_root(graph_root), self.structural_agg_leaf(graph_leaf)
            final_embedding.extend([root_emb, leaf_emb])
        return self.trans(torch.cat(final_embedding, dim=1))

    def compute_embedding(self, cascades):
        """aggregate the representations of users into a cascade embedding by
        structural learning and temporal learning"""
        # concat_state
        cas_history, cas_times, length = self.hgraph.get_cas_seq(cascades)
        cas_history = pad_sequence(cas_history, batch_first=True).to(self.device)
        """set all position&time emb"""
        cas_times = pad_sequence(cas_times, batch_first=True).to(self.device)
        cas_time_emb = self.time_position_encoder(cas_times.reshape(-1)).reshape(*cas_times.shape, -1)
        single_range = list(range(cas_times.shape[1]))
        cas_pos = torch.tensor([single_range] * cas_times.shape[0]).to(self.device)
        cas_pos_emb = self.position_embedding(cas_pos)
        """set graph feature"""
        graph_root, graph_leaf = None, None
        if self.use_structural:
            graph_root, graph_leaf = self.hgraph.get_cas_graph(cascades)
            graph_root, graph_leaf = graph_root.to(self.device), graph_leaf.to(self.device)
            graph_root.edata['time'] = self.time_position_encoder(graph_root.edata['time'])
            graph_leaf.edata['time'] = self.time_position_encoder(graph_leaf.edata['time'])
        if self.use_dynamic:
            dynamic_emb = self.compute_dynamic_emb(cascades, cas_history, length, cas_time_emb, cas_pos_emb, graph_root,
                                                   graph_leaf)
        if self.use_static:
            static_emb = self.compute_static_emb(cascades, cas_history, length, cas_time_emb, cas_pos_emb, graph_root,
                                                 graph_leaf)
        if self.use_static and self.use_dynamic:
            return (static_emb, dynamic_emb)
        elif self.use_static:
            return static_emb
        elif self.use_dynamic:
            return dynamic_emb


def get_embedding_module(module_type: str, dynamic_state: Mapping[str, DynamicState],
                         input_dimension: int, embedding_dimension: int, device: torch.device,
                         dropout: float = 0.1, hgraph: HGraph = None, max_time: float = 1.0,
                         time_num: int = 20, use_static: bool = False, user_num: int = -1,
                         max_global_time: float = 100.0, global_time_num: int = 20,
                         use_dynamic: bool = True, use_temporal: bool = True,
                         use_structural: bool = True) -> EmbeddingModule:
    if module_type == "identity":
        return IdentityEmbedding(dynamic_state=dynamic_state,
                                 embedding_dimension=embedding_dimension,
                                 device=device,
                                 dropout=dropout,
                                 hgraph=hgraph)
    elif module_type == 'aggregate':
        return AggregateEmbedding(dynamic_state=dynamic_state,
                                  embedding_dimension=embedding_dimension,
                                  device=device,
                                  dropout=dropout,
                                  hgraph=hgraph,
                                  input_dimension=input_dimension,
                                  max_time=max_time,
                                  time_num=time_num,
                                  use_static=use_static, user_num=user_num, max_global_time=max_global_time,
                                  global_time_num=global_time_num, use_dynamic=use_dynamic, use_temporal=use_temporal,
                                  use_structural=use_structural)
    elif module_type == 'concat':
        return ConcatEmbedding(dynamic_state=dynamic_state, embedding_dimension=embedding_dimension,
                               device=device, dropout=dropout, hgraph=hgraph,
                               max_global_time=max_global_time,
                               global_time_num=global_time_num)
    else:
        raise ValueError("Embedding Module {} not supported".format(module_type))
