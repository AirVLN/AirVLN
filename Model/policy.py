from typing import Any
import tqdm

from torch import nn as nn
import torch
import gc

from src.common.param import args
from Model.utils.common import CategoricalNet, CustomFixedCategorical


class ILPolicy(nn.Module):
    #
    def __init__(self, net, dim_actions):
        super().__init__()

        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )

    #
    def forward(self, *x):
        raise NotImplementedError

    #
    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        step=0,
    ):
        if args.policy_type in ['seq2seq', 'cma']:
            features, rnn_hidden_states = self.net(
                observations, rnn_hidden_states, prev_actions, masks
            )

        else:
            raise NotImplementedError

        distribution = self.action_distribution(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        return action, rnn_hidden_states

    def get_value(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def evaluate_actions(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    #
    def build_distribution(
        self, observations, rnn_hidden_states, prev_actions, masks
    ) -> CustomFixedCategorical:
        if args.policy_type in ['seq2seq', 'cma']:
            features, rnn_hidden_states = self.net(
                observations, rnn_hidden_states, prev_actions, masks
            )
        else:
            raise NotImplementedError

        return self.action_distribution(features)
