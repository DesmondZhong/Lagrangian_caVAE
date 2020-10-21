---

<div align="center">    
 
# Unsupervised Learning of Lagrangian Dynamics from Images for Prediction and Control

Yaofeng Desmond Zhong, Naomi Ehrich Leonard | 2020

[![Paper](http://img.shields.io/badge/paper-arxiv.2007.01926-B31B1B.svg)](https://arxiv.org/abs/2007.01926)
[![Conference](http://img.shields.io/badge/NeurIPS-2020-4b44ce.svg)](https://arxiv.org/abs/2007.01926)


</div>
 
This repository is the official implementation of [Unsupervised Learning of Lagrangian Dynamics from Images for Prediction and Control](https://arxiv.org/abs/2007.01926). 

## Requirements

This implementation is written with [PyTorch](https://pytorch.org/) and handles training with [PyTorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning), which makes our code easy to read and our results easy to reproduce. 

Coming Soon: Please install PyTorch according to the [official website](https://pytorch.org/get-started/locally/). To install all the other dependencies:

```bash
pip install -r requirements.txt
```

## Dataset
Coming Soon

## Training

To train the model, run this command:
```bash
# to train the pendulum example
python examples/pend_lag_cavae_trainer.py 
# to train the fully-actuated cartpole example
python examples/cart_lag_cavae_trainer.py 
# to train the fully-actuated acrobot example
python examples/acro_lag_cavae_trainer.py 
```
The commands above train the model on CPUs. Due to a [bug](https://github.com/pytorch/pytorch/issues/24823) in `torch.nn.functional.grid_sample`, you might encounter a segmentation fault if you train the model on GPUs. This bug has not been fixed in the latest PyTorch version (1.6.0) when this work is done. 

However, I successfully trained the pendulum example on GPU without error. Thanks to PyTorch-Lightning, you can train it on GPU with 
```bash
python examples/pend_lag_cavae_trainer.py --gpus 1
```
## Evaluation
Coming Soon

## Results
Coming Soon

## Acknowledgement
This research has been supported in part by ONR grant \#N00014-18-1-2873 and by the School of Engineering and Applied Science at Princeton University through the generosity of William Addy ’82.
Yaofeng Desmond Zhong would like to thank Christine Allen-Blanchette, Shinkyu Park, Sushant Veer and Anirudha Majumdar for helpful discussions. 
