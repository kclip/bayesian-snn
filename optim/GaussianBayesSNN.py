import torch
from torch.optim.optimizer import required
from optim.SNNSGD import snnsgd
from .optimizer import BayesOptimizer
from utils.misc import calculate_fan_in
import numpy as np


class GaussianBayesOptimizer(BayesOptimizer):
    def __init__(self, params, initial_params_mean, initial_params_prec,
                 prior_m, prior_s, fixed_prec=False,
                 weight_decay=0., lr=required, rho=required, device=required):

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, weight_decay=weight_decay, rho=rho,
                        fixed_prec=fixed_prec)
        super(GaussianBayesOptimizer, self).__init__(params, defaults)

        self.add_bayesian_param_group('mean', prior_m, initial_params_mean)
        self.add_bayesian_param_group('prec', prior_s, initial_params_prec)

        self.device = device

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for i, group in enumerate(self.param_groups):
            mean_params_with_grad = []
            grads = []
            grads_per_example = []
            prec_params_with_grad = []
            prior_s_list = []
            prior_m_list = []

            lr = group['lr']
            rho = group['rho']
            fixed_prec = group['fixed_prec']

            mean_params = group['mean']
            mean_priors = group['priors_mean']

            prec_params = group['prec']
            prec_priors = group['priors_prec']

            params = [p for p in group['params'] if p.requires_grad]
            for j, p in enumerate(params):
                if p.grad is not None:
                    grads.append(p.grad)
                    mean_params_with_grad.append(mean_params[j])
                    prior_m_list.append(mean_priors[j])
                    prec_params_with_grad.append(prec_params[j])
                    prior_s_list.append(prec_priors[j])
                    if not fixed_prec:
                        grads_per_example.append(p.sample_grad)

            snnsgd(fixed_prec, mean_params_with_grad,
                   prec_params_with_grad, grads_per_example, grads, lr, rho,
                   prior_m_list, prior_s_list)

        return loss

    @torch.no_grad()
    def update_weights(self, howto='train'):
        if (howto == 'train') or (howto == 'mode'):
            for group in self.param_groups:
                params = [p for p in group['params'] if p.requires_grad]
                for p, mean in zip(params, group['mean']):
                    p.data = mean.detach().clone()
        else:
            for group in self.param_groups:
                params = [p for p in group['params'] if p.requires_grad]
                for p, mean, prec in zip(params,
                                         group['mean'],
                                         group['prec']):
                    mean = mean.detach().clone()
                    prec = prec.detach().clone()
                    fan_in = calculate_fan_in(prec)
                    p.data = torch.normal(mean,
                                          prec.pow(-0.5) / np.sqrt(fan_in))

    def update_priors(self):
        for group in self.param_groups:
            for mean, mean_prior in zip(group['mean'], group['priors_mean']):
                mean_prior.data = mean.detach().clone()

            for prec, prec_prior in zip(group['prec'], group['priors_prec']):
                prec_prior.data = prec.detach().clone()

    def update_rho(self, rho):
        for group in self.param_groups:
            group['rho'] = rho