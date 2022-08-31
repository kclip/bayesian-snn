"""
Adapted from https://github.com/cybertronai/autograd-hacks/blob/master/autograd_hacks.py
Library for extracting interesting quantites from autograd, see README.md
Not thread-safe because of module-level variables
Notation:
o: number of output classes (exact Hessian), number of Hessian samples (sampled Hessian)
n: batch-size
do: output dimension (output channels for convolution)
di: input dimension (input channels for convolution)
Oh, Ow: output height, output width (convolution)
Kh, Kw: kernel height, kernel width (convolution)
A, activations: inputs into current layer
B, backprops: backprop values (aka Lop aka Jacobian-vector product) observed at current layer
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

_supported_layers = ['Linear', 'Conv2d', 'Dense', 'Conv']  # Supported layer class types
_hooks_disabled: bool = False           # work-around for https://github.com/pytorch/pytorch/issues/25723
_enforce_fresh_backprop: bool = False   # global switch to catch double backprop errors on Hessian computation


def add_hooks(model: nn.Module) -> None:
    """
    Adds hooks to model to save activations and backprop values.
    The hooks will
    1. save activations into param.activations during forward pass
    2. append backprops to params.backprops_list during backward pass.
    Call "remove_hooks(model)" to disable this.
    Args:
        model:
    """

    global _hooks_disabled
    _hooks_disabled = False

    handles = []
    for layer in model.modules():
        if _layer_type(layer) in _supported_layers:
                if _layer_type(layer) in ['Dense', 'Conv']:
                    if not hasattr(layer, 'weight'):
                        continue
                if layer.weight.requires_grad:
                    handles.append(layer.register_forward_hook(_capture_activations))
                    handles.append(layer.register_full_backward_hook(_capture_backprops))
    model.__dict__.setdefault('autograd_hacks_hooks', []).extend(handles)


def remove_hooks(model: nn.Module) -> None:
    """
    Remove hooks added by add_hooks(model)
    """

    assert model == 0, "not working, remove this after fix " \
                       "to https://github.com/pytorch/pytorch/issues/25723"

    if not hasattr(model, 'autograd_hacks_hooks'):
        print("Warning, asked to remove hooks, but no hooks found")
    else:
        for handle in model.autograd_hacks_hooks:
            handle.remove()
        del model.autograd_hacks_hooks


def disable_hooks() -> None:
    """
    Globally disable all hooks installed by this library.
    """

    global _hooks_disabled
    _hooks_disabled = True


def enable_hooks() -> None:
    """the opposite of disable_hooks()"""

    global _hooks_disabled
    _hooks_disabled = False


def is_supported(layer: nn.Module) -> bool:
    """Check if this layer is supported"""

    return _layer_type(layer) in _supported_layers


def _layer_type(layer: nn.Module) -> str:
    return layer.__class__.__name__


def _capture_activations(layer: nn.Module, input: List[torch.Tensor],
                         output: torch.Tensor):
    """Save activations into layer.activations in forward pass"""

    if _hooks_disabled:
        return
    assert _layer_type(layer) in _supported_layers, \
        "Hook installed on unsupported layer, this shouldn't happen"
    setattr(layer, "activations", input[0].detach())


def _capture_backprops(layer: nn.Module, _input, output):
    """Append backprop to layer.backprops_list in backward pass."""
    global _enforce_fresh_backprop

    if _hooks_disabled:
        return

    if _enforce_fresh_backprop:
        assert not hasattr(layer, 'backprops_list'), \
            "Seeing result of previous backprop," \
            " use clear_backprops(model) to clear"
        _enforce_fresh_backprop = False

    if not hasattr(layer, 'backprops_list'):
        setattr(layer, 'backprops_list', [])
    layer.backprops_list.append(output[0].detach())


def clear_backprops(model: nn.Module) -> None:
    """Delete layer.backprops_list in every layer."""
    for layer in model.modules():
        if hasattr(layer, 'backprops_list'):
            del layer.backprops_list


def zero_sample_grad(model: nn.Module) -> None:
    """Delete layer.backprops_list in every layer."""
    for p in model.parameters():
        if hasattr(p, 'sample_grad'):
            p.sample_grad.zero_()


def compute_samplegrad(model: nn.Module, loss_type: str = 'mean') -> None:
    """
    Compute per-example gradients and save them under 'param.sample_grad'.
     Must be called after loss.backprop()
    Args:
        model:
        loss_type: either "mean" or "sum" depending whether backpropped loss
         was averaged or summed over batch
    """

    assert loss_type in ('sum', 'mean')
    for layer in model.modules():

        layer_type = _layer_type(layer)

        if (layer_type not in _supported_layers) \
                or (not hasattr(layer, 'weight')) \
                or (not layer.weight.requires_grad):
            continue

        assert hasattr(layer, 'activations'), \
            "No activations detected, run forward after add_hooks(model)"
        assert hasattr(layer, 'backprops_list'), \
            "No backprops detected, run backward after add_hooks(model)"

        assert len(layer.backprops_list) == 1, \
            "Multiple backprops detected," \
            " make sure to call clear_backprops(model)"

        if len(layer.activations) == 1:
            if not hasattr(layer.weight, 'sample_grad'):
                layer.weight.sample_grad = layer.weight.grad
            else:
                layer.weight.sample_grad.add_(layer.weight.grad)

            if layer.bias is not None:
                if not hasattr(layer.bias, 'sample_grad'):
                    layer.bias.sample_grad = layer.bias.grad
                else:
                    layer.bias.sample_grad.add_(layer.bias.grad)

        A = layer.activations
        n = A.shape[0]
        if loss_type == 'mean':
            B = layer.backprops_list[0] * n
        else:  # loss_type == 'sum':
            B = layer.backprops_list[0]

        if layer_type == 'Linear':
            if not hasattr(layer.weight, 'sample_grad'):
                layer.weight.sample_grad = \
                    torch.einsum('ni,nj->nij', B, A).data
            else:
                layer.weight.sample_grad.add_(torch.einsum('ni,nj->nij',
                                                           B, A).data)
            if layer.bias is not None:
                if not hasattr(layer.bias, 'sample_grad'):
                    layer.bias.sample_grad = B.data
                else:
                    layer.bias.sample_grad.add_(B.data)

        elif layer_type == 'Conv2d':
            A = torch.nn.functional.unfold(A, layer.kernel_size,
                                           dilation=layer.dilation,
                                           padding=layer.padding,
                                           stride=layer.stride)
            B = B.reshape(n, -1, A.shape[-1])

            sample_grad = torch.einsum('ijk,ilk->ijl', B, A)

            shape = [n] + list(layer.weight.shape)
            if not hasattr(layer.weight, 'sample_grad'):
                setattr(layer.weight, 'sample_grad',
                        sample_grad.reshape(shape).clone())
            else:
                layer.weight.sample_grad.add_(sample_grad.reshape(shape).clone())

            if layer.bias is not None:
                if not hasattr(layer.bias, 'sample_grad'):
                    setattr(layer.bias, 'sample_grad',
                            torch.sum(B, dim=2).clone())
                else:
                    layer.bias.sample_grad.add_(torch.sum(B, dim=2).clone())

        elif layer_type == 'Dense':
            assert layer.kernel_size[-1] == 1, \
                "This layer is not compatible with 2d convolution backward"
            old_shape = A.shape
            A = torch.nn.functional.unfold(A.reshape(old_shape[0], -1, 1,
                                                     old_shape[-1]),
                                           layer.kernel_size[:2],
                                           dilation=layer.dilation[:2],
                                           padding=layer.padding[:2],
                                           stride=layer.stride[:2])
            B = B.reshape(n, -1, A.shape[-1])

            sample_grad = torch.einsum('ijk,ilk->ijl', B, A)

            shape = [n] + list(layer.weight.shape)
            if not hasattr(layer.weight, 'sample_grad'):
                setattr(layer.weight, 'sample_grad',
                        sample_grad.reshape(shape).clone())
            else:
                if layer.weight.sample_grad.shape == shape:
                    layer.weight.sample_grad.add_(
                        sample_grad.reshape(shape).clone()
                    )
                else:
                    layer.weight.sample_grad = sample_grad.reshape(shape).clone()

            if layer.bias is not None:
                if not hasattr(layer.bias, 'sample_grad'):
                    setattr(layer.bias, 'sample_grad',
                            torch.sum(B, dim=2).clone())
                else:
                    bias_sample_grad = torch.sum(B, dim=2).clone()
                    if layer.bias.sample_grad.shape == bias_sample_grad.shape:
                        layer.bias.sample_grad.add_(bias_sample_grad)
                    else:
                        layer.bias.sample_grad = bias_sample_grad

        elif layer_type == 'Conv':
            sample_grad = 0
            for i in range(A.shape[-1]):
                A_i = torch.nn.functional.unfold(A[..., i],
                                                 layer.kernel_size[:2],
                                                 dilation=layer.dilation[:2],
                                                 padding=layer.padding[:2],
                                                 stride=layer.stride[:2])
                B_i = B[..., i].reshape(n, -1, A_i.shape[-1])

                sample_grad += torch.einsum('ijk,ilk->ijl', B_i, A_i)

            shape = [n] + list(layer.weight.shape)
            sample_grad = sample_grad.reshape(shape).clone()
            if not hasattr(layer.weight, 'sample_grad'):
                setattr(layer.weight, 'sample_grad', sample_grad)
            else:
                if layer.weight.sample_grad.shape == shape:
                    layer.weight.sample_grad.add_(sample_grad)
                else:
                    layer.weight.sample_grad = sample_grad

            if layer.bias is not None:
                if not hasattr(layer.bias, 'sample_grad'):
                    setattr(layer.bias, 'sample_grad',
                            torch.sum(B, dim=2).clone())
                else:
                    bias_sample_grad = torch.sum(B, dim=2).clone()
                    if layer.bias.sample_grad.shape == bias_sample_grad.shape:
                        layer.bias.sample_grad.add_(bias_sample_grad)
                    else:
                        layer.bias.sample_grad = bias_sample_grad
