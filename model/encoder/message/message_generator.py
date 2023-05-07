import numpy as np
import torch.nn as nn
import torch
from collections import defaultdict
from typing import Dict, Mapping, Tuple, List, Any
from model.encoder.state.dynamic_state import DynamicState
from model.encoder.message.message_function import get_message_function
from model.encoder.message.message_aggregator import get_message_aggregator
from model.time_encoder import get_time_encoder


class MessageGenerator(nn.Module):
    def __init__(self, state: Mapping[str, DynamicState], time_encoder: Mapping[str, nn.Module], device: torch.device,
                 message_aggregator_type: str, message_function: Mapping[str, Any]):
        super(MessageGenerator, self).__init__()
        self.message_function = message_function
        self.message_aggregator = get_message_aggregator(message_aggregator_type, device)
        self.state = state
        self.time_encoder = time_encoder
        self.device = device

    def get_message(self, source_nodes: np.ndarray, destination_nodes: np.ndarray, trans_cascades: np.ndarray,
                    edge_times: torch.Tensor, relative_times: torch.Tensor, target: str) -> Tuple[Dict, Dict, Dict]:
        """
        Given a batch of interactions, first generate the message for each interaction, and then generate the unique
        message for each node by aggregating all messages of a node (a node may occur in multiple interactions,
        so it may have multiple messages)
        :param source_nodes: the sending users' id, ndarray of shape (batch)
        :param destination_nodes: the receiving users' id, ndarray of shape (batch)
        :param trans_cascades: the cascades' id, ndarray of shape (batch)
        :param edge_times: the happening timestamps of the interactions, tensor of shape (batch)
        :param relative_times: the time duration since the publication times of cascades in the interactions,
               tensor of shape (batch)
        :param target: what type of message should be generated, 'user' for users,
                       'cascade' for cascades, 'all' for both
        :return: the unique messages of nodes, which is a tuple of node ids, generated messages,
                 timestamps of interactions
        """
        ...

    def aggregate_transform(self, multi_nodes: Dict[str, List[int]],
                            multi_messages: Dict[str, Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]]) -> \
            Tuple[Dict[str, List[int]], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Aggregate messages for nodes
        :param multi_nodes: a dictionary, where multi_nodes[ntype] is a list that stores all nodes that should generate
                            a unique message in this batch
        :param multi_messages: a dictionary, where multi_messages[ntype][i] is a list that stores all the messages
                               of node `i` belonging to node type `ntype`
        :return: the unique messages of nodes, which is a tuple of node ids, generated messages,
                 timestamps of interactions
        """
        node_types = set(multi_nodes.keys())
        unique_multi_nodes, unique_multi_messages, unique_multi_timestamps = {}, {}, {}
        for node_type in node_types:
            unique_nodes, unique_messages, unique_timestamps = self.message_aggregator.aggregate(
                multi_nodes[node_type], multi_messages[node_type])
            unique_multi_nodes[node_type] = unique_nodes
            unique_multi_messages[node_type] = unique_messages
            unique_multi_timestamps[node_type] = unique_timestamps
        return unique_multi_nodes, unique_multi_messages, unique_multi_timestamps


class ConCatMessage(MessageGenerator):
    def __init__(self, state: Mapping[str, DynamicState], time_encoder: Mapping[str, nn.Module], device: torch.device,
                 message_aggregator_type: str, message_function: Mapping[str, Any]):
        super(ConCatMessage, self).__init__(state=state, time_encoder=time_encoder,
                                            device=device, message_aggregator_type=message_aggregator_type,
                                            message_function=message_function)

    def get_user_message(self, source_nodes: np.ndarray, destination_nodes: np.ndarray, trans_cascades: np.ndarray,
                         edge_times: torch.Tensor, unique_multi_nodes: Dict,
                         unique_multi_messages: Dict, unique_multi_timestamps: Dict):
        """generate messages for users
        :param source_nodes: the sending users' id, ndarray of shape (batch)
        :param destination_nodes: the receiving users' id, ndarray of shape (batch)
        :param trans_cascades: the cascades' id, ndarray of shape (batch)
        :param edge_times: the happening timestamps of the interactions, tensor of shape (batch)
        :param unique_multi_nodes: a dict to store the node id of each unique message
        :param unique_multi_messages: a dict to store the embedding vector of each unique message
        :param unique_multi_timestamps: a dict to store the timestamp of each unique message
        """
        raw_message = torch.cat(
            [self.state['user'].get_state(source_nodes, 'src'),
             self.state['user'].get_state(destination_nodes, 'dst'),
             self.state['cas'].get_state(trans_cascades)], dim=1)
        source_time_emb = self.time_encoder['user'](edge_times - self.state['user'].get_last_update(source_nodes))
        des_time_emb = self.time_encoder['user'](edge_times - self.state['user'].get_last_update(destination_nodes))
        source_message = self.message_function['user']['src'].compute_message(
            torch.cat([raw_message, source_time_emb], dim=1))
        dst_message = self.message_function['user']['dst'].compute_message(
            torch.cat([raw_message, des_time_emb], dim=1))
        nodes = {'src': [], 'dst': []}
        messages = {'src': defaultdict(list), 'dst': defaultdict(list)}
        for src_id, dst_id, src, dst, time in zip(source_nodes, destination_nodes, source_message, dst_message,
                                                  edge_times):
            nodes['src'].append(src_id)
            nodes['dst'].append(dst_id)
            messages['src'][src_id].append((src, time))
            messages['dst'][dst_id].append((dst, time))
        m_nodes, m_messages, m_times = self.aggregate_transform(nodes, messages)
        unique_multi_nodes['user'] = m_nodes
        unique_multi_messages['user'] = m_messages
        unique_multi_timestamps['user'] = m_times

    def get_cas_message(self, source_nodes: np.ndarray, destination_nodes: np.ndarray, trans_cascades: np.ndarray,
                        edge_times: torch.Tensor, pub_times: torch.Tensor, unique_multi_nodes: dict,
                        unique_multi_messages: dict, unique_multi_timestamps: dict):
        """generate messages for cascades
        :param source_nodes: the sending users' id, ndarray of shape (batch)
        :param destination_nodes: the receiving users' id, ndarray of shape (batch)
        :param trans_cascades: the cascades' id, ndarray of shape (batch)
        :param edge_times: the happening timestamps of the interactions, tensor of shape (batch)
        :param pub_times: the publication timestamps of cascades in the interactions, tensor of shape (batch)
        :param unique_multi_nodes: a dict to store the node id of each unique message
        :param unique_multi_messages: a dict to store the embedding vector of each unique message
        :param unique_multi_timestamps: a dict to store the timestamp of each unique message
        """
        nodes = {'cas': []}
        messages = {'cas': defaultdict(list)}
        raw_message = torch.cat(
            [self.state['user'].get_state(source_nodes, 'src'),
             self.state['user'].get_state(destination_nodes, 'dst'),
             self.state['cas'].get_state(trans_cascades)], dim=1)
        cas_time_emb = self.time_encoder['cas'](edge_times - pub_times)
        cas_message = self.message_function['cas'].compute_message(torch.cat([raw_message, cas_time_emb], dim=1))
        for cas_id, cas, time in zip(trans_cascades, cas_message, edge_times):
            nodes['cas'].append(cas_id)
            messages['cas'][cas_id].append((cas, time))
        m_nodes, m_messages, m_times = self.aggregate_transform(nodes, messages)
        unique_multi_nodes.update(m_nodes)
        unique_multi_messages.update(m_messages)
        unique_multi_timestamps.update(m_times)

    def get_message(self, source_nodes, destination_nodes, trans_cascades, edge_times, pub_times, target):
        unique_multi_nodes, unique_multi_messages, unique_multi_timestamps = dict(), dict(), dict()
        if target == 'user' or target == 'all':
            self.get_user_message(source_nodes, destination_nodes, trans_cascades, edge_times, unique_multi_nodes,
                                  unique_multi_messages, unique_multi_timestamps)
        if target == 'cas' or target == 'all':
            self.get_cas_message(source_nodes, destination_nodes, trans_cascades, edge_times, pub_times,
                                 unique_multi_nodes, unique_multi_messages, unique_multi_timestamps)
        return unique_multi_nodes, unique_multi_messages, unique_multi_timestamps


def set_message_function(single: bool, raw_message_dim: Dict[str, int], message_dim: int) -> \
        Mapping[str, Any]:
    cas_mf = get_message_function('mlp', raw_message_dim['cas'], message_dim)
    if single:
        user_src_mf = cas_mf
        user_dst_mf = cas_mf
    else:
        user_src_mf = get_message_function('mlp', raw_message_dim['user'], message_dim)
        user_dst_mf = get_message_function('mlp', raw_message_dim['user'], message_dim)
    user_message_function = nn.ModuleDict({'src': user_src_mf, 'dst': user_dst_mf})
    message_function = nn.ModuleDict({'user': user_message_function, 'cas': cas_mf})
    return message_function


def get_message_generator(generator_type: str, state: Mapping[str, DynamicState], time_encoder: Mapping[str, nn.Module],
                          time_dim: int, message_dim: int, node_feature_dim: int, device: torch.device,
                          message_aggregator_type: str, single: bool = False,
                          max_time: float = None) -> MessageGenerator:
    raw_message_dim = {'user': 3 * node_feature_dim + time_dim, 'cas': 3 * node_feature_dim + time_dim}
    message_function = set_message_function(single, raw_message_dim, message_dim)
    if generator_type == 'concat':
        return ConCatMessage(state=state, time_encoder=time_encoder,
                             device=device, message_aggregator_type=message_aggregator_type,
                             message_function=message_function)
    else:
        raise ValueError(f'No Implement generator type {generator_type}')
