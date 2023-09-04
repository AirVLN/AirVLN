from gym import spaces
import numpy as np
from typing import Dict
from pathlib import Path

import torch
from torch import Tensor
from torch import distributed as distrib
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from Model.utils import ddppo_resnet_utils as resnet
from utils.logger import logger
from src.common.param import args


class RunningMeanAndVar(nn.Module):
    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, n_channels, 1, 1))
        self.register_buffer("_var", torch.zeros(1, n_channels, 1, 1))
        self.register_buffer("_count", torch.zeros(()))
        self._mean: torch.Tensor = self._mean
        self._var: torch.Tensor = self._var
        self._count: torch.Tensor = self._count

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            n = x.size(0)
            # We will need to do reductions (mean) over the channel dimension,
            # so moving channels to the first dimension and then flattening
            # will make those faster.  Further, it makes things more numerically stable
            # for fp16 since it is done in a single reduction call instead of
            # multiple
            x_channels_first = (
                x.transpose(1, 0).contiguous().view(x.size(1), -1)
            )
            new_mean = x_channels_first.mean(-1, keepdim=True)
            new_count = torch.full_like(self._count, n)

            if distrib.is_initialized():
                distrib.all_reduce(new_mean)
                distrib.all_reduce(new_count)
                new_mean /= distrib.get_world_size()

            new_var = (
                (x_channels_first - new_mean).pow(2).mean(dim=-1, keepdim=True)
            )

            if distrib.is_initialized():
                distrib.all_reduce(new_var)
                new_var /= distrib.get_world_size()

            new_mean = new_mean.view(1, -1, 1, 1)
            new_var = new_var.view(1, -1, 1, 1)

            m_a = self._var * (self._count)
            m_b = new_var * (new_count)
            M2 = (
                m_a
                + m_b
                + (new_mean - self._mean).pow(2)
                * self._count
                * new_count
                / (self._count + new_count)
            )

            self._var = M2 / (self._count + new_count)
            self._mean = (self._count * self._mean + new_count * new_mean) / (
                self._count + new_count
            )

            self._count += new_count

        inv_stdev = torch.rsqrt(
            torch.max(self._var, torch.full_like(self._var, 1e-2))
        )
        # This is the same as
        # (x - self._mean) * inv_stdev but is faster since it can
        # make use of addcmul and is more numerically stable in fp16
        return torch.addcmul(-self._mean * inv_stdev, x, inv_stdev)


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        baseplanes: int = 32,
        ngroups: int = 32,
        spatial_size: int = 128,
        make_backbone=None,
        normalize_visual_inputs: bool = False,
    ):
        super().__init__()

        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
            spatial_size = observation_space.spaces["rgb"].shape[0] // 2
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
            spatial_size = observation_space.spaces["depth"].shape[0] // 2
        else:
            self._n_input_depth = 0

        if normalize_visual_inputs:
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(
                self._n_input_depth + self._n_input_rgb
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        if not self.is_blind:
            input_channels = self._n_input_depth + self._n_input_rgb
            self.backbone = make_backbone(input_channels, baseplanes, ngroups)

            final_spatial = int(
                spatial_size * self.backbone.final_spatial_compress
            )
            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(after_compression_flat_size / (final_spatial ** 2))
            )
            self.compression = nn.Sequential(
                nn.Conv2d(
                    self.backbone.final_channels,
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(True),
            )

            self.output_shape = (
                num_compression_channels,
                final_spatial,
                final_spatial,
            )

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        if self.is_blind:
            return None

        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = (
                rgb_observations.float() / 255.0
            )  # normalize RGB
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]

            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)

            cnn_input.append(depth_observations)

        x = torch.cat(cnn_input, dim=1)
        x = F.avg_pool2d(x, 2)

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x


class VlnResnetDepthEncoder(nn.Module):
    #
    def __init__(
        self,
        observation_space,
    ):
        super().__init__()

        if args.policy_type in ['seq2seq']:
            output_size = 128; self._output_size = output_size
            checkpoint = str(Path(args.project_prefix) / 'DATA/models/ddppo-models/gibson-2plus-resnet50.pth')
            backbone = "resnet50"
            resnet_baseplanes = 32
            normalize_visual_inputs = False
            trainable = False
            spatial_output = False
        elif args.policy_type in ['cma']:
            output_size = 128; self._output_size = output_size
            checkpoint = str(Path(args.project_prefix) / 'DATA/models/ddppo-models/gibson-2plus-resnet50.pth')
            backbone = "resnet50"
            resnet_baseplanes = 32
            normalize_visual_inputs = False
            trainable = False
            spatial_output = True
        else:
            raise NotImplementedError

        self.visual_encoder = ResNetEncoder(
            spaces.Dict({"depth": observation_space.spaces["depth"]}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)

        if checkpoint != "NONE":
            ddppo_weights = torch.load(checkpoint, map_location=torch.device('cpu'))

            weights_dict = {}
            for k, v in ddppo_weights["state_dict"].items():
                split_layer_name = k.split(".")[2:]
                if split_layer_name[0] != "visual_encoder":
                    continue

                layer_name = ".".join(split_layer_name[1:])
                weights_dict[layer_name] = v

            del ddppo_weights
            self.visual_encoder.load_state_dict(weights_dict, strict=True)

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    np.prod(self.visual_encoder.output_shape), output_size
                ),
                nn.ReLU(True),
            )
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1]
                * self.visual_encoder.output_shape[2],
                64,
            )

            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)

    #
    @property
    def output_size(self):
        return self._output_size

    #
    def forward(self, observations):
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        if "depth_features" in observations:
            x = observations["depth_features"]
        else:
            x = self.visual_encoder(observations)

        if self.spatial_output:
            b, c, h, w = x.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([x, spatial_features], dim=1)
        else:
            return self.visual_fc(x)


class TorchVisionResNet50(nn.Module):
    r"""
    Takes in observations and produces an embedding of the rgb component.

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
        device: torch.device
    """

    #
    def __init__(
        self,
        observation_space,
        device,
    ):
        super().__init__()

        if args.policy_type in ['seq2seq']:
            output_size = 256; self._output_size = output_size
            spatial_output = False
        elif args.policy_type in ['cma']:
            output_size = 256; self._output_size = output_size
            spatial_output = True
        else:
            raise NotImplementedError

        self.device = device
        self.resnet_layer_size = 2048
        linear_layer_input_size = 0

        self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        obs_size_0 = observation_space.spaces["rgb"].shape[0]
        obs_size_1 = observation_space.spaces["rgb"].shape[1]
        if obs_size_0 != 224 or obs_size_1 != 224:
            logger.warn(
                "TorchVisionResNet50: observation size is not conformant to expected ResNet input size [3x224x224]"
            )
        linear_layer_input_size += self.resnet_layer_size

        if self.is_blind:
            self.cnn = nn.Sequential()
            return

        self.cnn = models.resnet50(pretrained=True)

        # disable gradients for resnet, params frozen
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.eval()

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.fc = nn.Linear(linear_layer_input_size, output_size)
            self.activation = nn.ReLU()
        else:

            class SpatialAvgPool(nn.Module):
                def forward(self, x):
                    x = F.adaptive_avg_pool2d(x, (4, 4))

                    return x

            self.cnn.avgpool = SpatialAvgPool()
            self.cnn.fc = nn.Sequential()

            self.spatial_embeddings = nn.Embedding(4 * 4, 64)

            self.output_shape = (
                self.resnet_layer_size + self.spatial_embeddings.embedding_dim,
                4,
                4,
            )

        self.layer_extract = self.cnn._modules.get("avgpool")

    #
    @property
    def is_blind(self):
        return self._n_input_rgb == 0

    #
    @property
    def output_size(self):
        return self._output_size

    #
    def forward(self, observations):
        r"""Sends RGB observation through the TorchVision ResNet50 pre-trained
        on ImageNet. Sends through fully connected layer, activates, and
        returns final embedding.
        """

        def resnet_forward(observation):
            resnet_output = torch.zeros(
                1, dtype=torch.float32, device=self.device
            )

            def hook(m, i, o):
                resnet_output.set_(o)

            # output: [BATCH x RESNET_DIM]
            h = self.layer_extract.register_forward_hook(hook)
            self.cnn(observation)
            h.remove()
            return resnet_output

        if "rgb_features" in observations:
            resnet_output = observations["rgb_features"]
        else:
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT x WIDTH]
            rgb_observations = observations["rgb"].permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            resnet_output = resnet_forward(rgb_observations.contiguous())

        if self.spatial_output:
            b, c, h, w = resnet_output.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=resnet_output.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([resnet_output, spatial_features], dim=1)
        else:
            return self.activation(
                self.fc(torch.flatten(resnet_output, 1))
            )


class TorchVisionResNet50Place365(nn.Module):
    r"""
    Takes in observations and produces an embedding of the rgb component.
    Based on PLACE 365 Pretrained Model

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
        device: torch.device
    """

    #
    def __init__(
        self,
        observation_space,
        device,
    ):
        super().__init__()

        if args.policy_type in ['seq2seq']:
            output_size = 256; self._output_size = output_size
            spatial_output = False
        elif args.policy_type in ['cma']:
            output_size = 256; self._output_size = output_size
            spatial_output = True
        else:
            raise NotImplementedError

        self.device = device
        self.resnet_layer_size = 2048
        linear_layer_input_size = 0

        self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        obs_size_0 = observation_space.spaces["rgb"].shape[0]
        obs_size_1 = observation_space.spaces["rgb"].shape[1]
        if obs_size_0 != 224 or obs_size_1 != 224:
            logger.warn(
                "TorchVisionResNet50: observation size is not conformant to expected ResNet input size [3x224x224]"
            )
        linear_layer_input_size += self.resnet_layer_size

        if self.is_blind:
            self.cnn = nn.Sequential()
            return

        from scripts.get_feature import BuildModel_ResNet50_raw, model_param_convertor_raw
        place365_model = torch.load(
            str(Path(args.project_prefix) / 'DATA/models/resnet50/resnet50_places365.pth'),
            map_location=torch.device('cpu')
        )
        param = model_param_convertor_raw(place365_model['state_dict'])
        self.cnn = BuildModel_ResNet50_raw(param)

        # disable gradients for resnet, params frozen
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.eval()

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.fc = nn.Linear(linear_layer_input_size, output_size)
            self.activation = nn.ReLU()
        else:

            class SpatialAvgPool(nn.Module):
                def forward(self, x):
                    x = F.adaptive_avg_pool2d(x, (4, 4))

                    return x

            self.cnn.avgpool = SpatialAvgPool()
            self.cnn.fc = nn.Sequential()

            self.spatial_embeddings = nn.Embedding(4 * 4, 64)

            self.output_shape = (
                self.resnet_layer_size + self.spatial_embeddings.embedding_dim,
                4,
                4,
            )

        self.layer_extract = self.cnn._modules.get("avgpool")

    #
    @property
    def is_blind(self):
        return self._n_input_rgb == 0

    #
    @property
    def output_size(self):
        return self._output_size

    #
    def forward(self, observations):
        r"""Sends RGB observation through the TorchVision ResNet50 pre-trained
        on ImageNet. Sends through fully connected layer, activates, and
        returns final embedding.
        """

        def resnet_forward(observation):
            resnet_output = torch.zeros(
                1, dtype=torch.float32, device=self.device
            )

            def hook(m, i, o):
                resnet_output.set_(o)

            # output: [BATCH x RESNET_DIM]
            h = self.layer_extract.register_forward_hook(hook)
            self.cnn(observation)
            h.remove()
            return resnet_output

        if "rgb_features" in observations:
            resnet_output = observations["rgb_features"]
        else:
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT x WIDTH]
            rgb_observations = observations["rgb"].permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            resnet_output = resnet_forward(rgb_observations.contiguous())

        if self.spatial_output:
            b, c, h, w = resnet_output.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=resnet_output.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([resnet_output, spatial_features], dim=1)
        else:
            return self.activation(
                self.fc(torch.flatten(resnet_output, 1))
            )
