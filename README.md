# Code for Neurips2020 submission "Membership Inference with Privately Augmented Data Endorses the Benign while Suppresses the Adversary"

## Dependency

This code is tested with [torch 1.5](https://github.com/pytorch/pytorch) and [numpy 1.14](https://numpy.org/).

We use benchmark datasets [CIFAR10 and CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html). The program will download the dataset automatically at the first run.

## Training target model

We use random seed to generate individual transformation. Each  integer seed corresponds to a different transformation. Therefore, the random seeds chosen during training can be used to re-generate augmented instances that the model is trained on. 

The following command trains a ResNet110 model with 10 augmented instances for each image.
```
CUDA_VISIBLE_DEVICES=0 python cifar_train.py --arch resnet110 --aug_instances 10 --sess resnet110_N10
```

After standard training procedure, we record the outputs of trained model (loss and logits) in the results folder. 

You can also train with WRN16-8 and 2-layer ConvNet. The commands are listed below. 

```
CUDA_VISIBLE_DEVICES=0 python cifar_train.py --arch wrn16_8 --aug_instances 10 --weight_decay 5e-4  --sess wrn16_8_N10 

CUDA_VISIBLE_DEVICES=0 python cifar_train.py --arch convnet --aug_instances 10 --trainset_size 15000 --batchsize 256 --lr 0.01 --weight_decay 0.  --sess smallconv_N10
```

Set `aug_instance 0` will train the target model without data augmentation. You can train above models on CIFAR100 by adding `--c100` flag.

## Evaluating membership inference algorithms

We implement five MI algorithms in mi_attack.py. You can evaluate them simultaneously with a given session name.
```
python mi_attack.py --sess resnet110_N10 --aug_instances 10

python mi_attack.py --sess wrn16_8_c100_N10 --aug_instances 10 --c100
```

The `--random_t` flag allows one to evalute our algorithms with augmented data which is not used in training.

```
python mi_attack.py --sess resnet110_N10 --aug_instances 10 --random_t
```


The `--verify_unlearning` flag allows one to print the confidence of unlearning verifycation with given n_{i}. 

```
python mi_attack.py --sess resnet110_N10 --aug_instances 10 --verify_unlearning
```

