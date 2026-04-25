from typing import List, Optional

import torch


def grad(y: torch.Tensor, x: torch.Tensor, training: bool = True) -> torch.Tensor:
    """Compute gradient for scalar/vector (real/complex) outputs wrt x."""
    is_complex = y.is_complex()

    if y.dim() == 1:
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y)]
        grad_real = torch.autograd.grad(
            outputs=[y],
            inputs=[x],
            grad_outputs=grad_outputs,
            retain_graph=(training or is_complex),
            create_graph=training,
            allow_unused=True,
        )[0]
        assert grad_real is not None, "Real gradient is None"

        if is_complex:
            grad_imag = torch.autograd.grad(
                outputs=[y / 1j],
                inputs=[x],
                grad_outputs=grad_outputs,
                retain_graph=training,
                create_graph=training,
                allow_unused=True,
            )[0]
            assert grad_imag is not None, "Imag gradient is None"
            return grad_real + 1j * grad_imag
        return grad_real

    dim_y = y.shape[1]
    grad_outputs = [torch.ones_like(y[:, 0])]

    real_parts = []
    for i in range(dim_y):
        gi = torch.autograd.grad(
            outputs=[y[:, i]],
            inputs=[x],
            grad_outputs=grad_outputs,
            retain_graph=(training or i < dim_y - 1 or is_complex),
            create_graph=training,
            allow_unused=True,
        )[0]
        assert gi is not None, f"Real gradient for channel {i} is None"
        real_parts.append(gi)
    grad_real = torch.stack(real_parts, dim=2)

    if not is_complex:
        return grad_real

    imag_parts = []
    for i in range(dim_y):
        gi = torch.autograd.grad(
            outputs=[y[:, i] / 1j],
            inputs=[x],
            grad_outputs=grad_outputs,
            retain_graph=(training or i < dim_y - 1),
            create_graph=training,
            allow_unused=True,
        )[0]
        assert gi is not None, f"Imag gradient for channel {i} is None"
        imag_parts.append(gi)
    grad_imag = torch.stack(imag_parts, dim=2)
    return grad_real + 1j * grad_imag
