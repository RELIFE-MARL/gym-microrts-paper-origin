from typing import Optional

import torch
from torch import Tensor
from torch._six import nan
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property, logits_to_probs, probs_to_logits


class Categorical(Distribution):
    r"""
    Creates a categorical distribution parameterized by either :attr:`probs` or
    :attr:`logits` (but not both).

    .. note::
        It is equivalent to the distribution that :func:`torch.multinomial`
        samples from.

    Samples are integers from :math:`\{0, \ldots, K-1\}` where `K` is ``probs.size(-1)``.

    If `probs` is 1-dimensional with length-`K`, each element is the relative probability
    of sampling the class at that index.

    If `probs` is N-dimensional, the first N-1 dimensions are treated as a batch of
    relative probability vectors.

    .. note:: The `probs` argument must be non-negative, finite and have a non-zero sum,
              and it will be normalized to sum to 1 along the last dimension. attr:`probs`
              will return this normalized value.
              The `logits` argument will be interpreted as unnormalized log probabilities
              and can therefore be any real number. It will likewise be normalized so that
              the resulting probabilities sum to 1 along the last dimension. attr:`logits`
              will return this normalized value.

    See also: :func:`torch.multinomial`

    Example::

        >>> m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        >>> m.sample()  # equal probability of 0, 1, 2, 3
        tensor(3)

    Args:
        probs (Tensor): event probabilities
        logits (Tensor): event log probabilities (unnormalized)
    """
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    has_enumerate_support = True

    def __init__(
        self,
        probs: Optional[Tensor] = None,
        logits: Optional[Tensor] = None,
        validate_args=None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            if probs.dim() < 1:
                raise ValueError("`probs` parameter must be at least one-dimensional.")
            self.probs = probs / probs.sum(-1, keepdim=True)
        else:
            if logits.dim() < 1:
                raise ValueError("`logits` parameter must be at least one-dimensional.")
            # Normalize
            self.logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        self._param = self.probs if probs is not None else self.logits
        self._num_events = self._param.size()[-1]
        batch_shape = (
            self._param.size()[:-1] if self._param.ndimension() > 1 else torch.Size()
        )
        super(Categorical, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Categorical, _instance)
        batch_shape = torch.Size(batch_shape)
        param_shape = batch_shape + torch.Size((self._num_events,))
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(param_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(param_shape)
            new._param = new.logits
        new._num_events = self._num_events
        super(Categorical, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        return constraints.integer_interval(0, self._num_events - 1)

    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs)

    @lazy_property
    def probs(self):
        """Calculate Softmax of self.logits"""
        return logits_to_probs(self.logits)

    @property
    def param_shape(self):
        return self._param.size()

    @property
    def mean(self):
        return torch.full(
            self._extended_shape(),
            nan,
            dtype=self.probs.dtype,
            device=self.probs.device,
        )

    @property
    def variance(self):
        return torch.full(
            self._extended_shape(),
            nan,
            dtype=self.probs.dtype,
            device=self.probs.device,
        )

    def sample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        probs_2d = self.probs.reshape(-1, self._num_events)
        samples_2d = torch.multinomial(probs_2d, sample_shape.numel(), True).T
        return samples_2d.reshape(self._extended_shape(sample_shape))

    def log_prob(self, action: Tensor) -> Tensor:
        """Returns the log of the probability density/mass function evaluated at action

        Args:
            action (Tensor): of shape (num_envs * h * w) = (6144)

        Returns:
            Tensor: the same shape of action (6144)
        """
        if self._validate_args:
            self._validate_sample(action)
        action = action.long().unsqueeze(-1)  # of shape (6144, 1)

        # self.logits's shape can be either
        # tuple( (6144,6), (6144,4), (6144,4), (6144,4), (6144,4), (6144,7), (6144, 49) )
        # Therefore, action and log_pmf has the same shape as self.logits
        action, log_pmf = torch.broadcast_tensors(action, self.logits)

        # only keep the first dimension from (6144, x)
        action = action[..., :1]

        rs = log_pmf.gather(-1, action).squeeze(-1)
        return rs

    def entropy(self):
        min_real = torch.finfo(self.logits.dtype).min
        logits = torch.clamp(self.logits, min=min_real)
        p_log_p = logits * self.probs
        return -p_log_p.sum(-1)

    def enumerate_support(self, expand=True):
        num_events = self._num_events
        values = torch.arange(num_events, dtype=torch.long, device=self._param.device)
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        if expand:
            values = values.expand((-1,) + self._batch_shape)
        return values


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
