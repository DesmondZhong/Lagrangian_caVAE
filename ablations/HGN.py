"""
This implementation of HGN is based on the original HGN paper,
with the following exception.

The original paper use symplectic integrator for better accuracy
in long term prediction. However, our work focus on not only 
prediction but control as well. With a applied controller, the 
energy of the system is not conserved any more. Thus, we use RK4
for integration. Since this is the integrator we use to generate
our training data, we believe this is a fair choice. 

We only use trajectories without any control to train 
the HGN, since HGN does not model control. 

This implementation is based on the following implementation of 
Progressive GAN.
https://github.com/odegeasslbc/Progressive-GAN-pytorch
"""
# Standard library imports
import os, sys
from argparse import ArgumentParser

# Third party imports
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchdiffeq import odeint
from torch.distributions.normal import Normal

# local application imports
from utils import arrange_data, from_pickle, my_collate, ImageDataset


class Res_Block(torch.nn.Module):
    def __init__(self, c=64):
        super(Res_Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        h = self.block(x)
        h += x
        return torch.sigmoid(h)


class HGN_Decoder(torch.nn.Module):
    def __init__(self,c=64, out_c=3):
        super(HGN_Decoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2d_0 = nn.Conv2d(16, c, 1)

        self.progression_8 = Res_Block()
        self.progression_16 = Res_Block()
        self.progression_32 = Res_Block()

        self.to_rgb_8 = nn.Conv2d(c, out_c, 1)
        self.to_rgb_16 = nn.Conv2d(c, out_c, 1)
        self.to_rgb_32 = nn.Conv2d(c, out_c, 1)


        self.res_block_1 = Res_Block(c=c)
        self.res_block_2 = Res_Block(c=c)
        self.res_block_3 = Res_Block(c=c)
        self.max_step = 3

    def output(self, feat1, feat2, module1, module2, alpha):
        if 0 <= alpha < 1:
            skip_rgb = self.upsample(module1(feat1))
            out = (1-alpha) * skip_rgb + alpha * module2(feat2)
        else:
            out = module2(feat2)
        return torch.sigmoid(out)

    def forward(self, q, step=0, alpha=-1):
        if step > self.max_step:
            step = self.max_step
        # assume q has dimension 16, 4, 4
        out_8 = self.progression_8(self.conv2d_0(self.upsample(q))) # 16, 4,4 -> 16, 8, 8 -> 64, 8, 8
        if step == 1:
            return torch.sigmoid(self.to_rgb_8(out_8))

        out_16 = self.progression_16(self.upsample(out_8))
        if step == 2:
            return self.output(out_8, out_16, self.to_rgb_8, self.to_rgb_16, alpha)
        
        out_32 = self.progression_32(self.upsample(out_16))
        if step == 3:
            return self.output(out_16, out_32, self.to_rgb_16, self.to_rgb_32, alpha)

class ODEfunc(torch.nn.Module):
    def __init__(self):
        super(ODEfunc, self).__init__()
        self.H_func = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.ReLU(), # 64, 4, 4
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(), # 64, 4, 4
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(), # 64, 4, 4
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(), # 64, 4, 4
            nn.Conv2d(64, 64, 3),
            nn.ReLU(), # 64, 2, 2
            nn.Conv2d(64, 64, 2), # bs, 64, 1, 1
            nn.Flatten(), 
            nn.Linear(64, 1)
        )

    def forward(self, t, x):
        H = self.H_func(x)
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        dHdq, dHdp = dH.split([16, 16], dim=1)
        return torch.cat([dHdp, -dHdq], dim=1)
        

class Model(pl.LightningModule):

    def __init__(self, hparams, data_path=None):
        super(Model, self).__init__()
        self.hparams = hparams
        self.data_path = data_path
        self.T_pred = self.hparams.T_pred
        self.loss_fn = torch.nn.MSELoss(reduction='none')

        self.HGN_enc = nn.Sequential(
            nn.Conv2d(self.hparams.out_c*(self.T_pred+1), 32, 3), # encoder network 
            nn.ReLU(), # 32, 30, 30
            nn.Conv2d(32, 64, 3),
            nn.ReLU(), # 64, 28, 28
            nn.Conv2d(64, 64, 3),
            nn.ReLU(), # 64, 26, 26
            nn.Conv2d(64, 64, 3),
            nn.ReLU(), # 64, 24, 24
            nn.Conv2d(64, 64, 3),
            nn.ReLU(), # 64, 22, 22
            nn.Conv2d(64, 64, 3),
            nn.ReLU(), # 64, 20, 20
            nn.Conv2d(64, 64, 3),
            nn.ReLU(), # 64, 18, 18
            nn.Conv2d(64, 48, 3), # 48, 16, 16
        ) # bs, 3, 32, 32 -> bs, 48, 16, 16

        self.transform = nn.Sequential(
            nn.Conv2d(24, 64, 5), # encoder transformer network
            nn.ReLU(), # 64, 12, 12
            nn.Conv2d(64, 64, 5),
            nn.ReLU(), # 64, 8, 8
            nn.Conv2d(64, 32, 5), # 32, 4, 4
        )

        self.ode = ODEfunc()
        self.HGN_dec = HGN_Decoder(out_c=self.hparams.out_c)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='nearest')


    def train_dataloader(self):
        train_dataset = ImageDataset(self.data_path, self.hparams.T_pred, ctrl=False)
        # self.us = data['us']
        # self.u = self.us[self.idx]
        self.t_eval = torch.from_numpy(train_dataset.t_eval)
        return DataLoader(train_dataset, 
                          batch_size=self.hparams.batch_size, 
                          shuffle=True, 
                          collate_fn=my_collate,
                          drop_last=True,
                          num_workers=4)

    def on_batch_start(self, batch):
        # backward compatible to pl-0.8.5
        if self.global_step < 5000:
            self.step = 1
        if 5000 <= self.global_step < 10000:
            self.step = 2
        if 10000 <= self.global_step:
            self.step = 3
        rem = self.global_step % 5000
        self.alpha = min(1, rem / 2500)

    def forward(self, X):
        # encode
        [_, self.bs, _, self.d, self.d] = X.shape

        T = len(self.t_eval)
        z0_m_logv = self.HGN_enc(X.permute(1, 0, 2, 3, 4).reshape(self.bs, -1, self.d, self.d)) # bs, 48, 16, 16
        self.z0_m, self.z0_logv = z0_m_logv.split([24, 24], dim=1) # bs, 24, 16, 16
        self.Q_z0 = Normal(self.z0_m, self.z0_logv)
        self.P_normal = Normal(torch.zeros_like(self.z0_m), torch.ones_like(self.z0_logv))
        # reparametrize
        self.z0 = self.Q_z0.rsample()
        self.s0 = self.transform(self.z0)  # bs, 32, 4, 4

        self.sT = odeint(self.ode, self.s0, self.t_eval, method=self.hparams.solver)
        # T, bs, 32, 4, 4
        qT = self.sT[:,:,0:16].view(T*self.bs, 16, 4, 4)

        # decode
        self.Xrec = self.HGN_dec(qT, step=self.step, alpha=self.alpha) 
        if self.step == 1:
            self.Xrec = self.upsample_4(self.Xrec)
        if self.step == 2:
            self.Xrec = self.upsample_2(self.Xrec)
        self.Xrec = self.Xrec.view(T, self.bs, self.hparams.out_c, self.d, self.d)

    def training_step(self, train_batch, batch_idx):
        X, u = train_batch
        # make sure the channel dimension exist in the pendulum dataset
        if len(X.shape) == 4:
            X = X.view(X.shape[0], X.shape[1], 1, X.shape[2], X.shape[3])
        # if images are 64, 64, downsample to 32*32
        if X.shape[-1] == 64:
            [T, bs, c, _, _] = X.shape
            X = F.interpolate(X.view(T*bs, c, 64, 64), size=[32, 32]).view(T, bs, c, 32, 32)
        self.forward(X)

        lhood = - self.loss_fn(self.Xrec, X)
        lhood = lhood.sum([0, 2, 3, 4]).mean()
        kl_q = torch.distributions.kl.kl_divergence(self.Q_z0, self.P_normal).mean()

        loss = - lhood + kl_q

        logs = {'recon_loss': -lhood, 'kl_q_loss': kl_q, 'train_loss': loss, 
                'alpha': self.alpha, 'progressive_step': self.step}
        return {'loss':loss, 'log': logs, 'progress_bar': logs}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', default=1.5e-4, type=float)
        parser.add_argument('--batch_size', default=512, type=int)

        return parser