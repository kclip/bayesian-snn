import argparse
import torch
import numpy as np
import sys

from data.utils import make_twomoons_dataloader
from optim.utils import get_optimizer
import optim.SampleGradEngine as SampleGradEngine
from models.DECOLLEModels import Network
from utils.misc import make_experiment_dir
from utils.loss import DECOLLELoss
from utils.train import train_epoch_bayesian
from utils.test import test_bayesian

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    # Training arguments
    parser.add_argument('--home', default=r"\users\home", type=str)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_period', type=int, default=10)
    parser.add_argument('--num_ite', default=1, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--reg', default=0., type=float)
    parser.add_argument('--thr', default=1., type=float)
    parser.add_argument('--scale_grad', type=int, default=10000)
    parser.add_argument('--num_epochs', type=int, default=100)

    parser.add_argument('--num_samples_test', default=1, type=int)
    parser.add_argument('--rho', type=float, default=0.)
    parser.add_argument('--burn_in', type=int, default=10)

    parser.add_argument('--fixed_prec', action='store_true', default=False)
    parser.add_argument('--initial_prec', default=1., type=float)
    parser.add_argument('--prior_m', type=float, default=0.)
    parser.add_argument('--prior_s', default=1., type=float)

    parser.add_argument('--binary', action='store_true', default=False)
    parser.add_argument('--tau', default=1., type=float)
    parser.add_argument('--prior', default=0.5, type=float)

    parser.add_argument('--device', type=int, default=None)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.fixed_prec:
        args.thr = 64 * args.thr
        weight_scale = 64
        scale = 1
    else:
        weight_scale = 1
        scale = 1 << 6

    if args.binary:
        synapses = 'binary'
    else:
        synapses = 'real_valued'

    results_path = make_experiment_dir(args.home + '/results',
                                       'twomoons_bayesian_nepochs_%d'
                                       % args.num_epochs + synapses)

    with open(results_path + '/commandline_args.txt', 'w') as f:
        f.write('\n'.join(sys.argv[1:]))

    for ite in range(args.num_ite):
        acc_best = 0

        train_dl, train_dl_noshuffle, test_dl \
            = make_twomoons_dataloader(args.batch_size, results_path)

        net = Network(input_shape=20,
                      hidden_shape=[256, 256],
                      output_shape=1,
                      scale_grad=args.scale_grad,
                      thr=args.thr,
                      burn_in=args.burn_in,
                      thr_scaling=args.binary
                      ).to(device)
        SampleGradEngine.add_hooks(net)

        optimizer = get_optimizer(net, args, device,
                                  binary_synapses=args.binary)
        loss_fn = DECOLLELoss(torch.nn.BCEWithLogitsLoss(), net, args.reg)

        for epoch in range(args.num_epochs):
            print('Epoch %d / %d' % (epoch, args.num_epochs))

            train_iter = iter(train_dl)
            loss = train_epoch_bayesian(net, loss_fn,
                                        optimizer, train_iter,
                                        device, args.binary)

            if (epoch + 1) % args.test_period == 0:
                test_acc_mode, test_preds_mode, test_acc_ens, test_preds_ens, \
                test_acc_comm, test_preds_comm, _ \
                    = test_bayesian(net, test_dl,
                                    args.num_samples_test,
                                    optimizer, device)

                np.save(results_path + '/predictions_ens_latest_test.npy',
                        test_preds_ens.detach().numpy())
                np.save(results_path + '/predictions_comm_latest_test.npy',
                        test_preds_comm.detach().numpy())

                train_acc_mode, train_preds_mode, train_acc_ens, \
                train_preds_ens, train_acc_comm, train_preds_comm, true_labels \
                    = test_bayesian(net, train_dl_noshuffle,
                                    args.num_samples_test,
                                    optimizer, device)

                np.save(results_path + '/predictions_ens_latest_train.npy',
                        train_preds_ens.detach().numpy())
                np.save(results_path + '/predictions_comm_latest_train.npy',
                        train_preds_comm.detach().numpy())
