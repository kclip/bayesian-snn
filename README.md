# Bayesian spiking neural networks (SNNs)
Code for training SNNs via Bayesian learning using Intel's Lava framework (https://lava-nc.org/).
This code has been used for the following work:

N. Skatchkovsky, H. Jang, and O. Simeone, Bayesian Continual Learning via Spiking Neural Networks (https://arxiv.org/pdf/2208.13723.pdf)

# Running examples
Scripts for the paper are all in the `scripts` folder.
Example syntax to run the scripts are given in `launch_scripts.sh`.

Running scripts on the MNIST-DVS and DVSGestures dataset requires to use our
`neurodata` data preprocessing and loading package available at https://github.com/kclip/neurodata. 
 
# Dependencies
`lava-nc v.0.3.0`

`lava-dl v.0.2.0`

`numpy v.1.22.3`

`pytables v.3.6.1`

`scikit-learn v.1.1.0`

`torch v.1.11.0`

`tqdm v.4.64.0`


Author: Nicolas Skatchkovsky