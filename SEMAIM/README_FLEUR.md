Read me for running code, including the adjustments made to use the CIFAR dataset.

In main_pretrain.py file in the get_args_parses function, the datapath should be altered to the correct location of the cifar-10 dataset.
Further, good to note that currently the code is set to run on cpu, to change the device settings of cuda should be altered.

The specific versions of packages required are the following:
`Pytorch==1.13.0` and `torchvision==0.14.0` with `CUDA==11.6`
and pip install timm==0.4.5

Further only standard packages are required.

After this, the main_pretrain.py file should be run.