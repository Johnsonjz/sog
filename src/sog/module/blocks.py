from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Dense(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Optional[Union[Callable, nn.Module]] = nn.Identity(),
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))


def build_mlp(
    n_in: int,
    n_out: int,
    n_hidden: Optional[Union[int, Sequence[int]]] = None,
    n_layers: int = 2,
    activation: Callable = F.silu,
    bias: bool = True,
) -> nn.Module:
    if n_hidden is None:
        c_neurons = n_in
        n_neurons = []
        for _ in range(n_layers):
            n_neurons.append(c_neurons)
            c_neurons = max(n_out, c_neurons // 2)
        n_neurons.append(n_out)
    else:
        if isinstance(n_hidden, int):
            n_hidden = [n_hidden] * (n_layers - 1)
        n_neurons = [n_in] + list(n_hidden) + [n_out]

    layers = [
        Dense(n_neurons[i], n_neurons[i + 1], activation=activation, bias=bias)
        for i in range(n_layers - 1)
    ]
    layers.append(Dense(n_neurons[-2], n_neurons[-1], activation=None, bias=bias))
    return nn.Sequential(*layers)
