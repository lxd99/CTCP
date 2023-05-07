from torch import nn
import torch
import numpy as np
from model.encoder.state.dynamic_state import DynamicState
from typing import Dict, Mapping, List, Sequence, Type


class StateUpdater(nn.Module):
    def update_state(self, unique_multi_node_ids: Dict, unique_multi_messages: Dict, unique_multi_timestamps: Dict,
                     type: str = 'all'):
        """
        Given the ids and messages of nodes, update their dynamic states by different strategies
        :param unique_multi_node_ids: id of given nodes
        :param unique_multi_messages: messages of the given nodes
        :param unique_multi_timestamps: interaction timestamps of given nodes
        :param type: what type of nodes should be updated
        """
        ...

class SequenceStateUpdater(StateUpdater):
    def __init__(self, state: Mapping[str, DynamicState], message_dimension: int, state_dimension: int,
                 device: torch.device, ntypes: set, updater_function: Type[nn.RNNCellBase],
                 single_updater: bool = False):
        super(SequenceStateUpdater, self).__init__()
        self.state = state
        self.message_dimension = message_dimension
        self.state_dimension = state_dimension
        self.ntypes = ntypes
        self.device = device
        self.updaters = nn.ModuleDict()

        cas_updater = updater_function(input_size=message_dimension, hidden_size=state_dimension)
        if single_updater:
            user_src_updater = cas_updater
            user_dst_updater = cas_updater
        else:
            user_src_updater = updater_function(input_size=message_dimension, hidden_size=state_dimension)
            user_dst_updater = updater_function(input_size=message_dimension, hidden_size=state_dimension)
        user_updater = nn.ModuleDict({'src': user_src_updater, 'dst': user_dst_updater})
        self.updaters.update({'cas': cas_updater, 'user': user_updater})

    def update_src_dst_user_state(self, user_nodes: Dict[str, List[int]], user_messages: Dict[str, torch.Tensor],
                                  user_timestamps: Dict[str, torch.Tensor]):
        """update the dynamic states of users"""
        previous_timestamps = {'src': self.state['user'].get_last_update(user_nodes['src']),
                               'dst': self.state['user'].get_last_update(user_nodes['dst'])}
        for ntype in ['src', 'dst']:
            unique_node_ids = user_nodes[ntype]
            timestamps = user_timestamps[ntype]
            self.check_illegal(unique_node_ids, previous_timestamps[ntype], timestamps, 'user')
            last_time = self.state['user'].get_last_update(unique_node_ids)
            self.state['user'].set_last_update(unique_node_ids, torch.max(last_time, timestamps))
            state = self.state['user'].get_state(unique_node_ids, ntype)
            updated_state = self.updaters['user'][ntype](user_messages[ntype], state)
            self.state['user'].set_state(unique_node_ids, updated_state, ntype, set_cache=True)

    def update_cas_state(self, unique_node_ids: List[int], unique_messages: torch.Tensor, timestamps: torch.Tensor):
        """update the dynamic states of cascades"""
        self.check_illegal(unique_node_ids, self.state['cas'].get_last_update(unique_node_ids), timestamps, 'cas')
        state = self.state['cas'].get_state(unique_node_ids)
        self.state['cas'].set_last_update(unique_node_ids, timestamps)
        updated_state = self.updaters['cas'](unique_messages, state)
        self.state['cas'].set_state(unique_node_ids, updated_state, set_cache=True)

    def check_illegal(self, unique_node_ids: List[int], last_update_time: torch.Tensor, timestamps: torch.Tensor,
                      ntype: str):
        """check whether violate the timing constraints"""
        try:
            assert (last_update_time <= timestamps).all().item(), "Trying to " \
                                                                  "update state to time in the past"
        except Exception as e:
            unique_node_ids = torch.tensor(unique_node_ids)
            ill_ids = last_update_time > timestamps
            print(ntype, unique_node_ids[ill_ids], '\n',
                  last_update_time[ill_ids], '\n', timestamps[ill_ids])
            exit(0)

    def update_state(self, unique_multi_node_ids, unique_multi_messages, unique_multi_timestamps, type='all'):
        if type == 'user' or type == 'all':
            self.update_src_dst_user_state(unique_multi_node_ids['user'], unique_multi_messages['user'],
                                           unique_multi_timestamps['user'])
        if type == 'cas' or type == 'all':
            self.update_cas_state(unique_multi_node_ids['cas'], unique_multi_messages['cas'],
                                  unique_multi_timestamps['cas'])


def get_state_updater(module_type: str, state: Mapping[str, DynamicState], message_dimension: int, state_dimension: int,
                      device: torch.device, single_updater: bool, ntypes: set) -> StateUpdater:
    if module_type == "gru":
        return SequenceStateUpdater(state, message_dimension, state_dimension, device, ntypes, nn.GRUCell,
                                    single_updater)
    elif module_type == "rnn":
        return SequenceStateUpdater(state, message_dimension, state_dimension, device, ntypes, nn.RNNCell,
                                    single_updater)
