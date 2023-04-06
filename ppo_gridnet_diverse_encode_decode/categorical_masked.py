from typing import Optional

import torch
from torch import Tensor
from torch.distributions.categorical import Categorical


class CategoricalMasked(Categorical):
    def __init__(
        self,
        masks: Optional[Tensor],
        logits: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device
        self.masks = masks
        if len(self.masks) == 0:
            # if there is no masks available
            super(CategoricalMasked, self).__init__(logits=logits)
        else:
            # dtype = torch.bool
            self.masks = masks.bool()

            # set logits to negative infinity when invalid action (mask=False)
            logits = torch.where(
                self.masks,  # condition
                logits,  # where condition is True
                torch.tensor(-1e8, device=self.device),  # where condition is False
            )
            super(CategoricalMasked, self).__init__(logits=logits)

    def entropy(self) -> Tensor:
        """Calculate the Shannon entropy
        H = -sum(pk * log(pk))

        @Override the default entropy

        Returns:
            Tensor: Shannon entropy
        """
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()

        # If there are masks available,
        # we need to mask out the ones that are invalid

        # probs = F.softmax(logits, dim=-1)
        # elementwise multiplication
        # Ex: p_log_p is of shape (6144, 6)
        p_log_p = self.logits * self.probs

        # set p_log_p to 0.0 when invalid action
        p_log_p: Tensor = torch.where(
            self.masks,  # condition
            p_log_p,  # where condition is True
            torch.tensor(0.0).to(self.device),  # where condition is False
        )

        # Ex: return is of shape (6144)
        return -p_log_p.sum(-1)
