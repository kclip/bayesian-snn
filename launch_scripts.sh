#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --partition=nmes_simeone
#SBATCH --job-name=mnistdvs-continual
#SBATCH --time=120:00:00

module load cuda/11.4.2-gcc-9.4.0

### Two-moons
# Bayesian real-valued
python -m scripts.bayesian.offline.twomoons_decolle --lr=0.01 --home=/scratch/users/k1804053 --rho=0.01 --fixed_prec
# Bayesian binary
python -m scripts.bayesian.offline.twomoons_decolle --lr=0.1 --home=/scratch/users/k1804053 --rho=0.0001 --scale_grad=1000000 --binary
# Frequentist real-valued
python -m scripts.frequentist.offline.twomoons_decolle --lr=0.001 --home=/scratch/users/k1804053 --fixed_prec
# Frequentist binary
python -m scripts.frequentist.offline.twomoons_decolle --lr=0.001 --home=/scratch/users/k1804053 --binary

### DVS-Gestures
# Bayesian real-valued
python -m scripts.bayesian.offline.dvsgestures_decolle --lr=0.02 --home=/scratch/users/k1804053 --rho=0.001 --scale_grad=1000000 --fixed_prec
# Bayesian binary
python -m scripts.bayesian.offline.dvsgestures_decolle --lr=0.004 --home=/scratch/users/k1804053 --rho=0.001 --scale_grad=1000000 --binary
# Frequentist real-valued
python -m scripts.frequentist.offline.dvsgestures_decolle --lr=0.001 --home=/scratch/users/k1804053 --fixed_prec
# Frequentist binary
python -m scripts.frequentist.offline.dvsgestures_decolle --lr=0.00002 --home=/scratch/users/k1804053 --binary

### MNIST-DVS
# Bayesian real-valued
python -m scripts.bayesian.continual.mnistdvs_decolle --lr=0.5 --home=/scratch/users/k1804053 --rho=0.00001 --scale_grad=90000 --with_coresets --fixed_prec
# Bayesian binary
python -m scripts.bayesian.continual.mnistdvs_decolle --lr=0.1 --home=/scratch/users/k1804053 --rho=0.000004 --scale_grad=10000 --with_coresets --binary
# Frequentist real-valued
python -m scripts.frequentist.continual.mnistdvs_decolle --lr=0.003 --home=/scratch/users/k1804053 --test_period=10 --with_coresets --fixed_prec
# Frequentist binary
python -m scripts.frequentist.continual.mnistdvs_decolle --lr=0.0000006 --home=/scratch/users/k1804053 --test_period=10 --with_coresets --binary

### MNIST
# Bayesian real-valued
python -m scripts.bayesian.continual.mnist_decolle --lr=0.5 --home=/scratch/users/k1804053 --rho=0.000075 --scale_grad=100000 --with_coresets --fixed_prec
# Frequentist real-valued
python -m scripts.frequentist.continual.mnist_decolle --lr=0.0003 --home=/scratch/users/k1804053 --with_coresets --fixed_prec