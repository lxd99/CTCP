import torch
from torch import nn


class MessageFunction(nn.Module):
    """
    Module which computes the message for a given interaction.
    """

    def compute_message(self, raw_messages: torch.Tensor):
        """generate message for an interaction"""
        return None


class MLPMessageFunction(MessageFunction):
    def __init__(self, raw_message_dimension: int, message_dimension: int):
        super(MLPMessageFunction, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(raw_message_dimension, raw_message_dimension // 2),
            nn.ReLU(),
            nn.Linear(raw_message_dimension // 2, message_dimension),
            nn.ReLU(),
        )

    def compute_message(self, raw_messages: torch.Tensor) -> torch.Tensor:
        """
        :param raw_messages: raw message, tensor of shape (batch,raw_message_dim)
        :return message: returned message, tensor of shape (batch,message_dim)
        """
        messages = self.mlp(raw_messages)
        return messages


class IdentityMessageFunction(MessageFunction):
    def compute_message(self, raw_messages: torch.Tensor) -> torch.Tensor:
        """
        :param raw_messages: raw message, tensor of shape (batch,raw_message_dim)
        :return message: returned message, tensor of shape (batch,raw_message_dim)
        """
        return raw_messages


def get_message_function(module_type: str, raw_message_dimension: int, message_dimension: int) -> MessageFunction:
    if module_type == "mlp":
        return MLPMessageFunction(raw_message_dimension, message_dimension)
    elif module_type == "identity":
        return IdentityMessageFunction()
    else:
        raise NotImplementedError(f"Not message function type {module_type}")
