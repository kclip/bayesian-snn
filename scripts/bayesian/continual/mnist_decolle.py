import argparse
import torch
import numpy as np
import sys
from itertools import islice, chain, tee

from models.DECOLLEModels import Network
import optim.SampleGradEngine as SampleGradEngine
from utils.test import test_bayesian, compute_ece
from utils.misc import make_experiment_dir
from utils.loss import DECOLLELoss
from utils.train import train_epoch_bayesian
from optim.utils import get_optimizer
from data.utils import make_mnist_dataloader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    # Training arguments
    parser.add_argument('--home', default=r"\users\home", type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_period', type=int, default=1)
    parser.add_argument('--num_ite', default=3, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--reg', default=0., type=float)
    parser.add_argument('--thr', default=1.25, type=float)
    parser.add_argument('--scale_grad', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--T', type=int, default=40)

    parser.add_argument('--with_coresets', action='store_true', default=False)
    parser.add_argument('--coreset_length', type=int, default=15,
                        help='Number of batches in coreset')

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
                                       'mnist_bayesian_decolle_nepochs_%d_'
                                       % args.num_epochs + synapses)

    with open(results_path + '/commandline_args.txt', 'w') as f:
        f.write('\n'.join(sys.argv[1:]))

    # Create dataloaders
    tasks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    digits_all = [item for sublist in tasks for item in sublist]

    for ite in range(args.num_ite):
        tasks_seen = []
        accs_mode_per_task = [[] for _ in tasks]
        accs_ens_per_task = [[] for _ in tasks]
        accs_comm_per_task = [[] for _ in tasks]
        ece_ens_per_task = [[] for _ in tasks]
        ece_comm_per_task = [[] for _ in tasks]

        acc_best = 0
        net = Network(input_shape=28 * 28,
                      hidden_shape=[400],
                      output_shape=10,
                      scale_grad=args.scale_grad,
                      thr=args.thr,
                      burn_in=args.burn_in,
                      thr_scaling=args.binary).to(device)
        SampleGradEngine.add_hooks(net)

        optimizer = get_optimizer(net, args, device,
                                  binary_synapses=args.binary)
        loss_fn = DECOLLELoss(torch.nn.CrossEntropyLoss(), net, args.reg)

        for i, digits in enumerate(tasks):
            train_dl, test_dl_task \
                = make_mnist_dataloader(digits, args.batch_size, args.T)

            if len(tasks_seen) > 0:
                if args.with_coresets:
                    coresets = []
                    for task in tasks_seen:
                        train_dl_task_seen, _ \
                            = make_mnist_dataloader(task,
                                                    args.batch_size,
                                                    args.T)

                        train_iterator_seen = islice(iter(train_dl_task_seen),
                                                     args.coreset_length)
                        coreset_task = tee(train_iterator_seen, args.num_epochs)
                        coresets.append(coreset_task)


            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('Digits: ' + str(digits))
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

            for epoch in range(args.num_epochs):
                print('Epoch %d / %d' % (epoch + 1, args.num_epochs))

                if (len(tasks_seen) > 0) and args.with_coresets:
                    train_iterator = iter(train_dl)
                    for coreset in coresets:
                        train_iterator = chain(train_iterator, coreset[epoch])
                else:
                    train_iterator = iter(train_dl)

                loss = train_epoch_bayesian(net, loss_fn,
                                            optimizer, train_iterator,
                                            device, args.binary)

                if (epoch + 1) % args.test_period == 0:
                    # Get mode/ensemble/committee test acc on the current task
                    test_acc_mode_task, test_preds_mode_task, \
                    test_acc_ens_task, test_preds_ens_task, \
                    test_acc_comm_task, test_preds_comm_task, \
                    true_labels_test_task \
                        = test_bayesian(net, test_dl_task,
                                        args.num_samples_test,
                                        optimizer, device)
                    print('Mode acc at epoch %d for current task: %f'
                        % (epoch + 1, test_acc_mode_task))

                    print('ensemble acc at epoch %d for current task: %f' % (
                        epoch + 1, test_acc_ens_task))

                    print('committee acc at epoch %d for current task: %f' % (
                              epoch + 1, test_acc_comm_task))

                    ece_ens = compute_ece(test_preds_ens_task, 20,
                                          true_labels_test_task
                                          )
                    ece_comm = compute_ece(test_preds_comm_task, 20,
                                           true_labels_test_task
                                           )

                    accs_mode_per_task[i].append(test_acc_mode_task)
                    accs_ens_per_task[i].append(test_acc_ens_task)
                    accs_comm_per_task[i].append(test_acc_comm_task)
                    ece_ens_per_task[i].append(ece_ens)
                    ece_comm_per_task[i].append(ece_comm)

                    np.save(
                        results_path + '/test_preds_ensemble_task_%d_ite_%d.npy'
                        % (i + 1, ite), test_preds_ens_task.detach().numpy())
                    np.save(
                        results_path +'/test_preds_committee_task_%d_ite_%d.npy'
                        % (i + 1, ite), test_preds_comm_task.detach().numpy())
                    np.save(
                        results_path + '/true_labels_test_task_%d_ite_%d.npy'
                        % (i + 1, ite), true_labels_test_task.detach().numpy())

                    if len(tasks_seen) > 0:
                        print('Testing on previously seen digits...')
                        for j, task in enumerate(tasks_seen):
                            _, test_dl_seen_task \
                                = make_mnist_dataloader(task,
                                                        args.batch_size,
                                                        args.T)

                            test_acc_mode_task, test_preds_mode_task, \
                            test_acc_ens_task, test_preds_ens_task, \
                            test_acc_comm_task, test_preds_comm_task, \
                            true_labels_test_task \
                                = test_bayesian(net, test_dl_seen_task,
                                                args.num_samples_test,
                                                optimizer, device)

                            print('Mode acc at epoch %d for task %d: %f' % (
                                    epoch + 1, j, test_acc_mode_task))

                            print('ensemble acc at epoch %d for task %d: %f' % (
                                epoch + 1, j, test_acc_ens_task))

                            print('committee acc at epoch %d for task %d: %f' %
                                  (epoch + 1, j, test_acc_comm_task))

                            ece_ens = compute_ece(test_preds_ens_task, 20,
                                                  true_labels_test_task
                                                  )
                            ece_comm = compute_ece(test_preds_comm_task, 20,
                                                   true_labels_test_task
                                                   )

                            accs_mode_per_task[j].append(test_acc_mode_task)
                            accs_ens_per_task[j].append(test_acc_ens_task)
                            accs_comm_per_task[j].append(test_acc_comm_task)
                            ece_ens_per_task[j].append(ece_ens)
                            ece_comm_per_task[j].append(ece_comm)

                            np.save(
                                results_path + '/test_preds_ensemble_task_' +
                                               '%d_ite_%d.npy'
                                % (j + 1, ite),
                                test_preds_ens_task.detach().numpy())
                            np.save(
                                results_path + '/test_preds_committee_task' +
                                               '_%d_ite_%d.npy'
                                % (j + 1, ite),
                                test_preds_comm_task.detach().numpy())
                            np.save(
                                results_path + '/true_labels_test_task_' +
                                               '%d_ite_%d.npy'
                                % (j + 1, ite),
                                true_labels_test_task.detach().numpy())

                    for k in range(len(tasks_seen) + 1, len(tasks)):
                        accs_mode_per_task[k].append(0)
                        accs_ens_per_task[k].append(0)
                        accs_comm_per_task[k].append(0)
                        ece_ens_per_task[k].append(0)
                        ece_comm_per_task[k].append(0)

                    np.save(results_path + '/acc_mode_per_task_%d_ite_%d.npy'
                            % (i + 1, ite), np.array(accs_mode_per_task))
                    np.save(results_path + '/acc_ens_per_task_%d_ite_%d.npy'
                            % (i + 1, ite), np.array(accs_ens_per_task))
                    np.save(results_path + '/acc_comm_per_task_%d_ite_%d.npy'
                            % (i + 1, ite), np.array(accs_comm_per_task))
                    np.save(results_path + '/ece_ens_per_task_%d_ite_%d.npy'
                            % (i + 1, ite), np.array(ece_ens_per_task))
                    np.save(results_path + '/ece_comm_per_task_%d_ite_%d.npy'
                            % (i + 1, ite), np.array(ece_comm_per_task))

            optimizer.update_priors()
            tasks_seen += [digits]
