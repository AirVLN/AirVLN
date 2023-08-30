import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space

from Model.policy import ILPolicy
from Model.encoders.resnet_encoders import TorchVisionResNet50, TorchVisionResNet50Place365, VlnResnetDepthEncoder
from Model.encoders.instruction_encoder import InstructionEncoder, InstructionBertEncoder
from Model.encoders.rnn_state_encoder import build_rnn_state_encoder
from Model.aux_losses import AuxLosses

from src.common.param import args


class Seq2SeqPolicy(ILPolicy):
    #
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        out_model_config=None,
        device=torch.device("cpu"),
    ):
        super().__init__(
            Seq2SeqNet(
                observation_space=observation_space,
                num_actions=action_space.n,
                out_model_config=out_model_config,
                device=device,
            ),
            action_space.n,
        )

    #
    @classmethod
    def from_config(
        cls, observation_space: Space, action_space: Space, out_model_config=None,
        device=torch.device("cpu"),
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            out_model_config=out_model_config,
            device=device,
        )


class Seq2SeqNet(nn.Module):
    #
    def __init__(
        self, observation_space: Space, num_actions, out_model_config=None,
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.device = device

        # Init the instruction encoder 1
        if args.tokenizer_use_bert:
            self.instruction_encoder = InstructionBertEncoder()
        else:
            self.instruction_encoder = InstructionEncoder()

        # Init the depth encoder 2
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
        )

        # Init the RGB visual encoder 3
        if args.rgb_encoder_use_place365:
            self.rgb_encoder = TorchVisionResNet50Place365(
                observation_space, device,
            )
        else:
            self.rgb_encoder = TorchVisionResNet50(
                observation_space, device,
            )

        if args.SEQ2SEQ_use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        # Init the RNN state decoder 4
        rnn_input_size = (
            self.instruction_encoder.output_size
            + self.depth_encoder.output_size
            + self.rgb_encoder.output_size
        )

        if args.SEQ2SEQ_use_prev_action:
            rnn_input_size += self.prev_action_embedding.embedding_dim

        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=512,
            # num_layers=3,
        )

        self.progress_monitor = nn.Linear(
            self.state_encoder.hidden_size, 1
        )

        self._init_layers()

        self.train()

    #
    @property
    def output_size(self):
        return self.state_encoder.hidden_size

    #
    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    #
    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    #
    def _init_layers(self):
        nn.init.kaiming_normal_(
            self.progress_monitor.weight, nonlinearity="tanh"
        )
        nn.init.constant_(self.progress_monitor.bias, 0)

    #
    def forward(self, observations, rnn_states, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """

        if args.ablate_instruction:
            if self.instruction_encoder.config.final_state_only:
                instruction_embedding = torch.zeros(
                    size=(prev_actions.shape[0], self.instruction_encoder.output_size),
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                instruction_embedding = torch.zeros(
                    size=(prev_actions.shape[0], self.instruction_encoder.output_size, (observations["instruction"] != 0.0).long().sum(dim=1).max().detach().cpu().numpy().tolist()),
                    dtype=torch.float32,
                    device=self.device,
                )
        else:
            instruction_embedding = self.instruction_encoder(observations)

        if args.ablate_depth:
            depth_embedding = torch.zeros(
                size=[prev_actions.shape[0]] + list(self.depth_encoder.output_shape),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            depth_embedding = self.depth_encoder(observations)

        if args.ablate_rgb:
            rgb_embedding = torch.zeros(
                size=[prev_actions.shape[0]]+list(self.rgb_encoder.output_shape),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            rgb_embedding = self.rgb_encoder(observations)

        x = torch.cat(
            [instruction_embedding, depth_embedding, rgb_embedding], dim=1
        )

        if args.SEQ2SEQ_use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x = torch.cat([x, prev_actions_embedding], dim=1)

        x, rnn_states_out = self.state_encoder(x, rnn_states, masks)

        if args.PROGRESS_MONITOR_use and AuxLosses.is_active():
            progress_hat = torch.tanh(self.progress_monitor(x))
            progress_loss = F.mse_loss(
                progress_hat.squeeze(1),
                observations["progress"],
                reduction="none",
            )
            AuxLosses.register_loss(
                "progress_monitor",
                progress_loss,
                args.PROGRESS_MONITOR_alpha,
            )

        return x, rnn_states_out

