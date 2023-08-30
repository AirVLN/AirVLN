import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space

from Model.policy import ILPolicy
from Model.encoders.instruction_encoder import InstructionEncoder, InstructionBertEncoder
from Model.encoders.resnet_encoders import TorchVisionResNet50, TorchVisionResNet50Place365, VlnResnetDepthEncoder
from Model.encoders.rnn_state_encoder import build_rnn_state_encoder
from Model.aux_losses import AuxLosses
from Model.utils.CN import CN

from src.common.param import args


class CMAPolicy(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        out_model_config=None,
        device=torch.device("cpu"),
    ):
        super().__init__(
            CMANet(
                observation_space=observation_space,
                num_actions=action_space.n,
                out_model_config=out_model_config,
                device=device,
            ),
            action_space.n,
        )

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


class CMANet(nn.Module):
    r"""A cross-modal attention (CMA) network that contains:
    Instruction encoder
    Depth encoder
    RGB encoder
    CMA state encoder
    """

    def __init__(
        self, observation_space: Space, num_actions, out_model_config=None,
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.device = device

        model_config = CN.clone()
        model_config.STATE_ENCODER_hidden_size = 512
        model_config.STATE_ENCODER_rnn_type = 'GRU'
        model_config.PROGRESS_MONITOR_use = args.PROGRESS_MONITOR_use
        model_config.PROGRESS_MONITOR_alpha = args.PROGRESS_MONITOR_alpha
        self.model_config = model_config

        # Init the instruction encoder 1
        if args.tokenizer_use_bert:
            self.instruction_encoder = InstructionBertEncoder()
        else:
            self.instruction_encoder = InstructionEncoder()

        # Init the depth encoder 2
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
        )

        # Init the RGB encoder 3
        if args.rgb_encoder_use_place365:
            self.rgb_encoder = TorchVisionResNet50Place365(
                observation_space, device,
            )
        else:
            self.rgb_encoder = TorchVisionResNet50(
                observation_space, device,
            )

        self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        hidden_size = model_config.STATE_ENCODER_hidden_size
        self._hidden_size = hidden_size

        self.rgb_linear = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(
                self.rgb_encoder.output_shape[0],
                self.rgb_encoder.output_size,
            ),
            nn.ReLU(True),
        )
        self.depth_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                np.prod(self.depth_encoder.output_shape),
                self.depth_encoder.output_size,
            ),
            nn.ReLU(True),
        )

        # Init the RNN state decoder
        rnn_input_size = self.depth_encoder.output_size
        rnn_input_size += self.rgb_encoder.output_size
        rnn_input_size += self.prev_action_embedding.embedding_dim

        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER_hidden_size,
            rnn_type=model_config.STATE_ENCODER_rnn_type,
            num_layers=1,
        )

        self._output_size = (
            model_config.STATE_ENCODER_hidden_size
            + self.rgb_encoder.output_size
            + self.depth_encoder.output_size
            + self.instruction_encoder.output_size
        )

        self.rgb_kv = nn.Conv1d(
            self.rgb_encoder.output_shape[0],
            hidden_size // 2 + self.rgb_encoder.output_size,
            1,
        )

        self.depth_kv = nn.Conv1d(
            self.depth_encoder.output_shape[0],
            hidden_size // 2 + self.depth_encoder.output_size,
            1,
        )

        self.state_q = nn.Linear(hidden_size, hidden_size // 2)
        self.text_k = nn.Conv1d(
            self.instruction_encoder.output_size, hidden_size // 2, 1
        )
        self.text_q = nn.Linear(
            self.instruction_encoder.output_size, hidden_size // 2
        )

        self.register_buffer(
            "_scale", torch.tensor(1.0 / ((hidden_size // 2) ** 0.5))
        )

        self.second_state_compress = nn.Sequential(
            nn.Linear(
                self._output_size + self.prev_action_embedding.embedding_dim,
                self._hidden_size,
            ),
            nn.ReLU(True),
        )

        self.second_state_encoder = build_rnn_state_encoder(
            input_size=self._hidden_size,
            hidden_size=self._hidden_size,
            rnn_type=model_config.STATE_ENCODER_rnn_type,
            num_layers=1,
        )
        self._output_size = model_config.STATE_ENCODER_hidden_size

        self.progress_monitor = nn.Linear(self.output_size, 1)

        self._init_layers()

        self.train()

    @property
    def output_size(self):
        return self._output_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers + (
            self.second_state_encoder.num_recurrent_layers
        )

    def _init_layers(self):
        if self.model_config.PROGRESS_MONITOR_use:
            nn.init.kaiming_normal_(
                self.progress_monitor.weight, nonlinearity="tanh"
            )
            nn.init.constant_(self.progress_monitor.bias, 0)

    def _attn(self, q, k, v, mask=None):
        logits = torch.einsum("nc, nci -> ni", q, k)

        if mask is not None:
            logits = logits - mask.float() * 1e8

        attn = F.softmax(logits * self._scale, dim=1)

        return torch.einsum("ni, nci -> nc", attn, v)

    def forward(self, observations, rnn_states, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """

        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().view(-1)
        )  # [BATCH x 32]

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
        depth_embedding = torch.flatten(depth_embedding, 2)

        if args.ablate_rgb:
            rgb_embedding = torch.zeros(
                size=[prev_actions.shape[0]]+list(self.rgb_encoder.output_shape),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            rgb_embedding = self.rgb_encoder(observations)
        rgb_embedding = torch.flatten(rgb_embedding, 2)

        rgb_in = self.rgb_linear(rgb_embedding)
        depth_in = self.depth_linear(depth_embedding)

        state_in = torch.cat([rgb_in, depth_in, prev_actions], dim=1)
        rnn_states_out = rnn_states.detach().clone()
        (
            state,
            rnn_states_out[:, 0 : self.state_encoder.num_recurrent_layers],
        ) = self.state_encoder(
            state_in,
            rnn_states[:, 0 : self.state_encoder.num_recurrent_layers],
            masks,
        )

        text_state_q = self.state_q(state)
        text_state_k = self.text_k(instruction_embedding)
        text_mask = (instruction_embedding == 0.0).all(dim=1)
        text_embedding = self._attn(
            text_state_q, text_state_k, instruction_embedding, text_mask
        )

        rgb_k, rgb_v = torch.split(
            self.rgb_kv(rgb_embedding), self._hidden_size // 2, dim=1
        )
        depth_k, depth_v = torch.split(
            self.depth_kv(depth_embedding), self._hidden_size // 2, dim=1
        )

        text_q = self.text_q(text_embedding)
        rgb_embedding = self._attn(text_q, rgb_k, rgb_v)
        depth_embedding = self._attn(text_q, depth_k, depth_v)

        x = torch.cat(
            [
                state,
                text_embedding,
                rgb_embedding,
                depth_embedding,
                prev_actions,
            ],
            dim=1,
        )
        x = self.second_state_compress(x)
        (
            x,
            rnn_states_out[:, self.state_encoder.num_recurrent_layers :],
        ) = self.second_state_encoder(
            x,
            rnn_states[:, self.state_encoder.num_recurrent_layers :],
            masks,
        )

        if self.model_config.PROGRESS_MONITOR_use and AuxLosses.is_active():
            progress_hat = torch.tanh(self.progress_monitor(x))
            progress_loss = F.mse_loss(
                progress_hat.squeeze(1),
                observations["progress"],
                reduction="none",
            )
            AuxLosses.register_loss(
                "progress_monitor",
                progress_loss,
                self.model_config.PROGRESS_MONITOR_alpha,
            )

        return x, rnn_states_out

