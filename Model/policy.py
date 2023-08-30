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
        if args.policy_type in ['seq2seq', 'cma', 'unet', 'vlnbert']:
            features, rnn_hidden_states = self.net(
                observations, rnn_hidden_states, prev_actions, masks
            )

        elif args.policy_type in ['hcm']:
            features, rnn_hidden_states, stop_output = self.net(
                observations, rnn_hidden_states, prev_actions, masks
            )

        else:
            raise NotImplementedError

        distribution = self.action_distribution(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        if args.run_type == 'eval':
            action = action.view(args.maxAction, args.batchSize)
            action = action[step, :].view(-1, 1)

        return action, rnn_hidden_states

    def get_value(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def evaluate_actions(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    #
    def build_distribution(
        self, observations, rnn_hidden_states, prev_actions, masks
    ) -> CustomFixedCategorical:
        if args.policy_type in ['seq2seq', 'cma', 'unet']:
            features, rnn_hidden_states = self.net(
                observations, rnn_hidden_states, prev_actions, masks
            )
        elif args.policy_type in ['hcm']:
            features, rnn_hidden_states, stop_output = self.net(
                observations, rnn_hidden_states, prev_actions, masks
            )
        elif args.policy_type in ['vlnbert']:
            features = []
            # pbar = tqdm.tqdm(total=args.maxAction, desc='number of steps')
            for step_idx in range(args.maxAction):
                gc.collect()
                torch.cuda.empty_cache()
                feature, rnn_hidden_states = self.net(
                    observations, rnn_hidden_states, prev_actions, masks, step_idx
                )
                features.append(feature.cpu())
                # pbar.update()
            features = torch.stack(features, dim=0).to(self.net.device)
        else:
            raise NotImplementedError

        return self.action_distribution(features)
