import torch
from typing import List, Optional
from torch import Tensor


def snnsgd(fixed_prec, mean_params, prec_params, grads_per_example,
           grads, lr, rho, prior_m_list, prior_s_list):
    if fixed_prec:
        scale_list = [prec.detach().pow(-1) for prec in prec_params]
        snnsgd_fixed(mean_params,
                     grads,
                     lr,
                     rho,
                     scale_list,
                     prior_m_list)
    else:
        snnsgd_prec(prec_params,
                    grads_per_example,
                    lr,
                    rho,
                    prior_s_list
                    )

        scale_list = [prec.detach().pow(-1) for prec in prec_params]

        snnsgd_mean(mean_params,
                    grads,
                    lr,
                    scale_list,
                    rho,
                    prior_m_list,
                    prior_s_list
                    )


# Fixed prec
def snnsgd_fixed(params: List[Tensor],
                 grads: List[Tensor],
                 lr: float,
                 rho: float,
                 scale_list: List[float],
                 prior_m_list: List[Optional[Tensor]]):

    for i, param in enumerate(params):
        grad = grads[i]
        prior_m = prior_m_list[i]
        scale = scale_list[i]
        param.add_(scale * (grad - rho * prior_m), alpha=-lr)


# Prec learning
def snnsgd_mean(params: List[Tensor],
                grads: List[Tensor],
                lr: float,
                scale_list: List[float],
                rho: float,
                prior_m_list: List[Optional[Tensor]],
                prior_s_list: List[Optional[Tensor]]
                ):

    for i, param in enumerate(params):
        grad = grads[i]
        prior_m = prior_m_list[i]
        prior_s = prior_s_list[i]
        scale = scale_list[i]

        update = scale \
                 * (grad - rho * prior_s * (prior_m - param.clone().detach()))
        param.add_(update, alpha=-lr)


def snnsgd_prec(prec_list: List[Tensor],
                grads_per_example: List[Tensor],
                lr: float,
                rho: float,
                prior_s_list: List[Optional[Tensor]]
                ):

    for i, prec in enumerate(prec_list):
        grad_example = grads_per_example[i]
        prior_s = prior_s_list[i]
        hess = torch.mean(grad_example.pow(2), dim=0)
        prec.mul_(1 - lr * rho).add_(hess + rho * prior_s, alpha=lr)
