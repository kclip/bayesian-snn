import argparse
import torch
import numpy as np
import sys

from data.utils import make_dvsgestures_dataloader
from models.DECOLLEModels import Network
import optim.SampleGradEngine as SampleGradEngine
from optim.utils import get_optimizer
from utils.test import test_bayesian
from utils.train import train_epoch_bayesian
from utils.misc import make_experiment_dir
from utils.loss import DECOLLELoss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    # Training arguments
    parser.add_argument('--home', default=r"\users\home", type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_period', type=int, default=10)
    parser.add_argument('--num_ite', default=3, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--reg', default=0., type=float)
    parser.add_argument('--thr', default=1.25, type=float)
    parser.add_argument('--scale_grad', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=200)

    parser.add_argument('--num_samples_test', default=10, type=int)
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
        if args.device is not None:
            device = torch.device('cuda:%d' % args.device)
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
                                       'dvsgestures_bayesian_nepochs_%d_dataset_'
                                       % args.num_epochs + synapses)

    with open(results_path + '/commandline_args.txt', 'w') as f:
        f.write('\n'.join(sys.argv[1:]))

    for ite in range(args.num_ite):
        acc_best = 0
        net = Network(input_shape=2 * 32 * 32,
                      hidden_shape=[4096, 4096],
                      output_shape=11,
                      scale_grad=args.scale_grad,
                      thr=args.thr,
                      burn_in=args.burn_in,
                      thr_scaling=args.binary
                      ).to(device)
        SampleGradEngine.add_hooks(net)

        dataset_path = args.home \
                       + r"/datasets/DvsGesture/dvs_gestures_events_new.hdf5"
        test_dataset_path = args.home \
                            + r"/datasets/DvsGesture/dvs_gestures_in_distrib_test.hdf5"

        train_dl, test_dl, test_dl_indistrib \
            = make_dvsgestures_dataloader(dataset_path,
                                          test_dataset_path,
                                          args.batch_size)


        optimizer = get_optimizer(net, args, device,
                                  binary_synapses=args.binary)
        loss_fn = DECOLLELoss(torch.nn.CrossEntropyLoss(), net, args.reg)

        for epoch in range(args.num_epochs):
            print('Epoch %d / %d' % (epoch+1, args.num_epochs))

            train_iter = iter(train_dl)
            loss = train_epoch_bayesian(net, loss_fn,
                                        optimizer, train_iter,
                                        device, args.binary)

            if (epoch + 1) % args.test_period == 0:
                test_acc_mode, test_preds_mode, test_acc_ens, test_preds_ens, \
                test_acc_comm, test_preds_comm, true_labels_test \
                    = test_bayesian(net, test_dl, args.num_samples_test,
                                    optimizer, device)

                print('Mode acc at epoch %d: %f' % (epoch + 1, test_acc_mode))
                print('ensemble acc at epoch %d: %f' % (epoch+1, test_acc_ens))
                print('committee acc at epoch %d: %f' % (epoch+1, test_acc_comm))

                if test_acc_ens.numpy() >= acc_best:
                    acc_best = test_acc_ens.numpy()
                    np.save(results_path + '/test_preds_ensemble_ite_%d.npy'
                            % ite, test_preds_ens.detach().numpy())
                    np.save(results_path + '/test_preds_committee_ite_%d.npy'
                            % ite, test_preds_ens.detach().numpy())
                    np.save(results_path + '/true_labels_test.npy',
                            true_labels_test.detach().numpy())

                    test_acc_mode_id, test_preds_mode_id, test_acc_ens_id,\
                    test_preds_ens_id, test_acc_comm_id, test_preds_comm_id,\
                    true_labels_test_id = test_bayesian(net, test_dl,
                                                        args.num_samples_test,
                                                        optimizer, device)

                    np.save(results_path
                            + '/test_preds_ensemble_indistrib_ite_%d.npy' % ite,
                            test_preds_ens_id.detach().numpy())
                    np.save(results_path + '/true_labels_indistrib_test.npy',
                            true_labels_test_id.detach().numpy())
