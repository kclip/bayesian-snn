import argparse
import torch
import numpy as np
import sys

from data.utils import make_dvsgestures_dataloader
from models.DECOLLEModels import Network
from utils.train import train_epoch_frequentist
from utils.misc import make_experiment_dir
from utils.loss import DECOLLELoss
from utils.test import test_frequentist

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
    parser.add_argument('--tau_grad', type=float, default=0.03)
    parser.add_argument('--weight_scale', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--burn_in', type=int, default=10)

    parser.add_argument('--fixed_prec', action='store_true', default=False)
    parser.add_argument('--binary', action='store_true', default=False)

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
                                       'dvsgestures_freq_nepochs_%d_dataset_'
                                       % args.num_epochs + synapses)

    with open(results_path + '/commandline_args.txt', 'w') as f:
        f.write('\n'.join(sys.argv[1:]))

    if args.fixed_prec:
        args.thr = 64 * args.thr
        weight_scale = 64
        scale = 1
    else:
        weight_scale = 1
        scale = 1 << 6

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

        dataset_path = args.home \
                       + r"/datasets/DvsGesture/dvs_gestures_events_new.hdf5"
        test_dataset_path = args.home \
                            + r"/datasets/DvsGesture/dvs_gestures_in_distrib_test.hdf5"

        train_dl, test_dl, test_dl_indistrib \
            = make_dvsgestures_dataloader(dataset_path,
                                          test_dataset_path,
                                          args.batch_size)

        loss_fn = DECOLLELoss(torch.nn.CrossEntropyLoss(reduction='mean'),
                              net, reg=args.reg)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

        for epoch in range(args.num_epochs):
            print('Epoch %d / %d' % (epoch+1, args.num_epochs))
            train_iter = iter(train_dl)
            loss = train_epoch_frequentist(net, loss_fn, optimizer,
                                           train_iter, device)

            if (epoch + 1) % args.test_period == 0:
                test_acc, test_preds, true_labels_test \
                    = test_frequentist(net, test_dl, device)

                print('Test acc at epoch %d: %f' % (epoch + 1, test_acc))

                if test_acc.numpy() >= acc_best:
                    acc_best = test_acc.numpy()
                    np.save(results_path + '/test_preds_ite_%d.npy' % ite,
                            test_preds.detach().numpy())
                    np.save(results_path + '/true_labels_test.npy',
                            true_labels_test.detach().numpy())

                    if test_dl_indistrib is not None:
                        test_acc_indistrib, test_preds_indistrib,\
                        true_labels_test_indistrib \
                            = test_frequentist(net, test_dl_indistrib, device)

                        np.save(results_path
                                + '/test_preds_indistrib_ite_%d.npy' % ite,
                                test_preds_indistrib.detach().numpy())
                        np.save(results_path
                                + '/true_labels_indistrib_test.npy',
                                true_labels_test_indistrib.detach().numpy())
