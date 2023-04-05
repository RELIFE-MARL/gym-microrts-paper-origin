from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from .categorical_masked import CategoricalMasked
from .decoder import Decoder
from .encoder import Encoder
from .utils import layer_init
from .vec_monitor import VecMonitor


class Agent(nn.Module):
    def __init__(self, envs: VecMonitor, device: torch.device, mapsize=16 * 16):
        super(Agent, self).__init__()
        self.envs = envs
        self.device = device
        self.mapsize = mapsize
        h, w, c = envs.observation_space.shape

        self.encoder = Encoder(c)

        self.actor = Decoder(78)

        self.critic = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(256, 128), std=1),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)  # "bhwc" -> "bchw"

    def get_action(
        self,
        x: Tensor,
        action: Optional[Tensor] = None,
        invalid_action_masks: Optional[Tensor] = None,
        envs: Optional[VecMonitor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        logits: Tensor = self.actor(self.forward(x))
        grid_logits = logits.reshape(-1, envs.action_space.nvec[1:].sum())
        split_logits = torch.split(
            grid_logits, envs.action_space.nvec[1:].tolist(), dim=1
        )

        if action is None:
            invalid_action_masks = torch.tensor(
                np.array(envs.vec_client.getMasks(0))
            ).to(self.device)
            invalid_action_masks = invalid_action_masks.view(
                -1, invalid_action_masks.shape[-1]
            )
            split_invalid_action_masks = torch.split(
                invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(), dim=1
            )
            multi_categoricals = [
                CategoricalMasked(logits=logits, masks=iam, device=self.device)
                for (logits, iam) in zip(split_logits, split_invalid_action_masks)
            ]
            action = torch.stack(
                [categorical.sample() for categorical in multi_categoricals]
            )
        else:
            invalid_action_masks = invalid_action_masks.view(
                -1, invalid_action_masks.shape[-1]
            )
            action = action.view(-1, action.shape[-1]).T
            split_invalid_action_masks = torch.split(
                invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(), dim=1
            )
            multi_categoricals = [
                CategoricalMasked(logits=logits, masks=iam, device=self.device)
                for (logits, iam) in zip(split_logits, split_invalid_action_masks)
            ]
        logprob = torch.stack(
            [
                categorical.log_prob(a)
                for a, categorical in zip(action, multi_categoricals)
            ]
        )
        entropy = torch.stack(
            [categorical.entropy() for categorical in multi_categoricals]
        )
        num_predicted_parameters = len(envs.action_space.nvec) - 1
        logprob = logprob.T.view(-1, 256, num_predicted_parameters)
        entropy = entropy.T.view(-1, 256, num_predicted_parameters)
        action = action.T.view(-1, 256, num_predicted_parameters)
        invalid_action_masks = invalid_action_masks.view(
            -1, 256, envs.action_space.nvec[1:].sum() + 1
        )
        return (
            action,
            logprob.sum(1).sum(1),
            entropy.sum(1).sum(1),
            invalid_action_masks,
        )

    def get_value(self, x: Tensor) -> Tensor:
        return self.critic(self.forward(x))
