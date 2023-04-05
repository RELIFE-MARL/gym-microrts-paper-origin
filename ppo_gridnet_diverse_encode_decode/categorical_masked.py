from typing import Optional

import torch
from torch.distributions.categorical import Categorical


class CategoricalMasked(Categorical):
    def __init__(
        self,
        probs=None,
        logits=None,
        validate_args=None,
        masks=[],
        sw=None,
        device: Optional[torch.device] = None,
    ):
        self.device = device
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.bool()
            logits = torch.where(
                self.masks, logits, torch.tensor(-1e8, device=self.device)
            )
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(self.device))
        return -p_log_p.sum(-1)
