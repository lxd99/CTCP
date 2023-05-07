import torch.nn as nn
import torch
from typing import Union


class Linear(nn.Module):
    def __init__(self, emb_dim: int):
        super(Linear, self).__init__()
        self.lins = nn.Sequential(nn.Linear(emb_dim, emb_dim // 2),
                                  nn.ReLU(), nn.Linear(emb_dim // 2, 1))

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Map the cascade embeddings into popularity counts
        :param emb: tensor of shape (batch,emb_dim)
        :return: tensor of shape (batch)
        """
        return torch.squeeze(self.lins(emb), dim=1)


class MergeLinear(nn.Module):
    def __init__(self, emb_dim: int, prob: float):
        super(MergeLinear, self).__init__()
        self.prob = nn.Parameter(torch.tensor(prob), requires_grad=False)
        self.dynamic_fn = Linear(emb_dim)
        self.static_fn = Linear(emb_dim)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Map the cascade embeddings into popularity counts
        :param emb: tensor of shape (batch,emb_dim)
        :return: tensor of shape (batch)
        """
        static_emb, dynamic_emb = emb
        pred = self.prob * self.static_fn(static_emb) + (1 - self.prob) * self.dynamic_fn(dynamic_emb)
        return pred


def get_predictor(emb_dim: int, predictor_type: str = 'linear', merge_prob: float = 0.5) -> Union[Linear, MergeLinear]:
    if predictor_type == 'linear':
        return Linear(emb_dim)
    elif predictor_type == 'merge':
        return MergeLinear(emb_dim, merge_prob)
    else:
        raise ValueError('Not implemented predictor type!')
