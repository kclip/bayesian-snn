import torch


class DECOLLELoss(object):
    '''
    Adapted from https://github.com/nmi-lab/decolle-public
    Computes the DECOLLE Loss for the model, defined as the sum of the per-layer
    local pseudo-losses + a regularization term
    '''

    def __init__(self, loss_fn, net, reg=0):
        ''''
        loss_fn: loss function used for each layer
        net: model to optimize
        red: float:
        '''
        self.nlayers = len(net.readout_layers)
        self.loss_fn = loss_fn
        self.reg = reg

    def __len__(self):
        return self.nlayers

    def __call__(self, readouts, voltages, target):
        loss = 0

        for r, v in zip(readouts, voltages):
            for t in range(r.shape[-1]):
                loss += self.loss_fn(r[..., t], target) / r.shape[-1]
                if self.reg > 0.:
                    vflat = v.reshape(v.shape[0], -1)

                    loss += self.reg * torch.mean(torch.relu(vflat + .01))
                    loss += self.reg * 3e-3 \
                            * torch.mean(torch.relu(0.1 - torch.sigmoid(vflat)))
        return loss
