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

        # ignore the first part (position) = 78
        self.action_space_nvec_sum = envs.action_space.nvec[1:].sum()

        # ignore the first part (position) = [6, 4, 4, 4, 4, 7, 49]
        self.action_space_nvec_list = envs.action_space.nvec[1:].tolist()

        self.encoder = Encoder(c)

        self.actor = Decoder(self.action_space_nvec_sum)

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
        # of shape (num_envs/b, h, w, self.action_space_nvec_sum) = (24, 16, 16, 78)
        logits: Tensor = self.actor(self.forward(x))

        # of shape (num_envs * h * w, self.action_space_nvec_sum) = (6144, 78)
        grid_logits = logits.reshape(-1, self.action_space_nvec_sum)

        # of shape tuple( (6144,6), (6144,4), (6144,4), (6144,4), (6144,4), (6144,7), (6144, 49) )
        split_logits = torch.split(grid_logits, self.action_space_nvec_list, dim=1)

        if action is None:
            # of shape (num_envs, h, w, self.action_space_nvec_sum+1) = (24, 16, 16, 79)
            invalid_action_masks = torch.tensor(
                np.array(envs.vec_client.getMasks(0))
            ).to(self.device)

            # of shape (num_envs * h * w, self.action_space_nvec_sum+1) = (6144, 79)
            invalid_action_masks = invalid_action_masks.view(
                -1, invalid_action_masks.shape[-1]
            )

            # of shape tuple( (6144,6), (6144,4), (6144,4), (6144,4), (6144,4), (6144,7), (6144, 49) )
            split_invalid_action_masks = torch.split(
                invalid_action_masks[:, 1:],  # of shape (6144, 78)
                self.action_space_nvec_list,
                dim=1,
            )

            # Multi-Discrete Action
            multi_categoricals = [
                # each logits are
                CategoricalMasked(logits=logits, masks=iam, device=self.device)
                for (logits, iam) in zip(split_logits, split_invalid_action_masks)
            ]

            # Sample actions and stack them together. Of shape (7, 6144)
            action = torch.stack(
                [categorical.sample() for categorical in multi_categoricals]
            )
        else:
            # of shape (num_envs * h * w, self.action_space_nvec_sum+1) = (6144, 79)
            invalid_action_masks = invalid_action_masks.view(
                -1, invalid_action_masks.shape[-1]
            )

            # ?
            action = action.view(-1, action.shape[-1]).T

            # of shape tuple( (6144,6), (6144,4), (6144,4), (6144,4), (6144,4), (6144,7), (6144, 49) )
            split_invalid_action_masks = torch.split(
                invalid_action_masks[:, 1:], self.action_space_nvec_list, dim=1
            )

            # Multi-Discrete Action
            multi_categoricals = [
                CategoricalMasked(logits=logits, masks=iam, device=self.device)
                for (logits, iam) in zip(split_logits, split_invalid_action_masks)
            ]
        
        # 
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
        """Calculate scalar value based on the observation
        Flow: input -> encoder -> critic -> output

        Args:
            x (Tensor): Observation of shape (b, h, w, n_features)
                b can be num_envs

        Returns:
            Tensor: scalar value of the observation (b, )
        """
        return self.critic(self.forward(x))
