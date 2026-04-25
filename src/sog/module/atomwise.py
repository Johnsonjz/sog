from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Dense, build_mlp


class Atomwise(nn.Module):
    """Predict latent charges from per-atom descriptors."""

    def __init__(
        self,
        n_in: Optional[int] = None,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        bias: bool = True,
        activation: Callable = F.silu,
        add_linear_nn: bool = False,
        output_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.bias = bias
        self.activation = activation
        self.add_linear_nn = add_linear_nn
        self.output_scaling_factor = output_scaling_factor

        self.outnet: Optional[nn.Module] = None
        self.linear_nn: Optional[nn.Module] = None

        if self.n_in is not None:
            self._build_network(self.n_in)

    def _build_network(self, n_in: int) -> None:
        self.outnet = build_mlp(
            n_in=n_in,
            n_out=self.n_out,
            n_hidden=self.n_hidden,
            n_layers=self.n_layers,
            activation=self.activation,
            bias=self.bias,
        )
        if self.add_linear_nn:
            self.linear_nn = Dense(n_in, self.n_out, bias=self.bias, activation=None)

    def forward(self, desc: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        del batch
        if self.n_in is None:
            self.n_in = int(desc.shape[1])
            self._build_network(self.n_in)
            self.to(desc.device)
        else:
            assert self.n_in == int(desc.shape[1]), "Descriptor width changed at runtime"

        assert self.outnet is not None
        out = self.outnet(desc)
        if self.linear_nn is not None:
            out = out + self.linear_nn(desc)
        return out * self.output_scaling_factor
