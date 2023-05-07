import torch
import numpy as np
import torch.nn as nn
from typing import Union, Mapping, Dict


class TimeDifferenceEncoder(torch.nn.Module):
    def __init__(self, dimension: int):
        super(TimeDifferenceEncoder, self).__init__()
        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)

        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                           .float().reshape(dimension, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Mapping the time difference into a vector
        :param t: time difference, a tensor of shape (batch)
        :return: time vector, a tensor of shape (batch,dimension)
        """
        # t has shape [batch_size]
        t = t.unsqueeze(dim=1)
        output = torch.cos(self.w(t))
        return output


class TimeSlotEncoder(torch.nn.Module):
    def __init__(self, dimension: int, max_time: float, time_num: int):
        super(TimeSlotEncoder, self).__init__()
        self.max_time = max_time
        self.time_num = time_num
        self.emb = nn.Embedding(time_num, dimension)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Mapping a timestamp into a vector
        :param t: timestamp, a tensor of shape (batch)
        :return: time vector, a tensor of shape (batch,dimension)
        """
        t = (t / self.max_time * (self.time_num - 1)).to(torch.long)
        return self.emb(t)


def get_time_encoder(model_type: str, dimension: int, max_time: Dict[str, float] = None, time_num: int = 20,
                     single: bool = False) -> Mapping[str, Union[TimeDifferenceEncoder, TimeSlotEncoder]]:
    if model_type == 'difference':
        user_time_encoder = TimeDifferenceEncoder(dimension)
        if single:
            cas_time_encoder = user_time_encoder
        else:
            cas_time_encoder = TimeDifferenceEncoder(dimension)
        return nn.ModuleDict({'user': user_time_encoder, 'cas': cas_time_encoder})
    elif model_type == 'slot':
        return nn.ModuleDict({
            'user': TimeSlotEncoder(dimension, max_time['user'], time_num),
            'cas': TimeSlotEncoder(dimension, max_time['cas'], time_num)
        })
    else:
        raise ValueError("Not Implemented Model Type")
