# Unsupervised Learning of Visual Embeddings

This is a Pytorch re-implementation of the Local Aggregation algorithm.


### Prerequisites

* Pytorch 1.2.0
* Ubuntu 16.04
* [Faiss==1.6.1](https://github.com/facebookresearch/faiss)
* dotmap
* * tqdm
* tensorboardX

### Runtime Setup
```
source init_env.sh
```

### Model training

This implementation is designed to support ResNets trained with the LA algorithm, and we have validated it with ResNet-18. The LA algorithm necessitates a preliminary 10-epoch training phase with the IR algorithm as a warm start, which is initiated using the command below:
```
CUDA_VISIBLE_DEVICES=0 python scripts/instance.py ./config/imagenet_ir.json
```
Set instance_exp_dir in ./config/imagenet_la.json and execute the following command to start LA training:
```
CUDA_VISIBLE_DEVICES=0 python scripts/localagg.py ./config/imagenet_la.json
```
The default setting trains both IR and LA using a single GPU, though multi-GPU training is also available.


### Transfer learning 
Once LA training is finished, execute the following command to start the transfer learning process for ImageNet:
```
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py ./config/imagenet_ft.json
```
