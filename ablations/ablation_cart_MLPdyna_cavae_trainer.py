# Standard library imports
from argparse import ArgumentParser
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

# Third party imports
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torchdiffeq import odeint
from torch.distributions.normal import Normal

# local application imports
from lag_caVAE.lag import Lag_Net_R1_T1
from lag_caVAE.nn_models import MLP_Encoder, MLP, MLP_Decoder, PSD, MatrixNet
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform
from utils import arrange_data, from_pickle, my_collate, ImageDataset

seed_everything(0)


class Model(pl.LightningModule):

    def __init__(self, hparams, data_path=None):
        super(Model, self).__init__()
        self.hparams = hparams
        self.data_path = data_path
        self.T_pred = self.hparams.T_pred
        self.loss_fn = torch.nn.MSELoss(reduction='none')

        self.recog_net_1 = MLP_Encoder(64*64, 300, 2, nonlinearity='elu')
        self.recog_net_2 = MLP_Encoder(64*64, 300, 3, nonlinearity='elu')
        self.obs_net_1 = MLP_Decoder(1, 100, 64*64, nonlinearity='elu')
        self.obs_net_2 = MLP_Decoder(1, 100, 64*64, nonlinearity='elu')        

        g_net = MatrixNet(3, 100, 4, shape=(2,2))
        g_baseline_net = MLP(5, 400, 2)

        self.ode = Lag_Net_R1_T1(g_net=g_net,
                                 g_baseline=g_baseline_net,
                                 dyna_model='g_baseline')

    def train_dataloader(self):
        train_dataset = ImageDataset(self.data_path, self.hparams.T_pred, ctrl=True)
        # self.us = data['us']
        # self.u = self.us[self.idx]
        self.t_eval = torch.from_numpy(train_dataset.t_eval)
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True, collate_fn=my_collate)

    def angle_vel_est(self, q0_m_n, q1_m_n, delta_t):
        delta_cos = q1_m_n[:,0:1] - q0_m_n[:,0:1]
        delta_sin = q1_m_n[:,1:2] - q0_m_n[:,1:2]
        q_dot0 = - delta_cos * q0_m_n[:,1:2] / delta_t + delta_sin * q0_m_n[:,0:1] / delta_t
        return q_dot0

    def encode(self, batch_image):
        r_m_logv = self.recog_net_1(batch_image[:, 0].reshape(self.bs, self.d*self.d))
        r_m, r_logv = r_m_logv.split([1, 1], dim=1)
        r_m = torch.tanh(r_m) ; r_v = torch.exp(r_logv) + 0.0001
        theta = self.get_theta(1, 0, r_m[:, 0], 0)
        grid = F.affine_grid(theta, torch.Size((self.bs, 1, self.d, self.d)))
        pole_att_win = F.grid_sample(batch_image[:, 1:2], grid)
        phi_m_logv = self.recog_net_2(pole_att_win.reshape(self.bs, self.d*self.d))
        phi_m, phi_logv = phi_m_logv.split([2, 1], dim=1)
        phi_m_n = phi_m / phi_m.norm(dim=-1, keepdim=True) ; phi_v = F.softplus(phi_logv) + 1
        return r_m, r_v, phi_m, phi_v, phi_m_n

    def get_theta(self, cos, sin, x, y, bs=None):
        # x, y should have shape (bs, ) 
        bs = self.bs if bs is None else bs
        theta = torch.zeros([bs, 2, 3], dtype=self.dtype, device=self.device)
        theta[:, 0, 0] += cos ;  theta[:, 0, 1] += sin ; theta[:, 0, 2] += x
        theta[:, 1, 0] += -sin ; theta[:, 1, 1] += cos ; theta[:, 1, 2] += y
        return theta

    def get_theta_inv(self, cos, sin, x, y, bs=None):
        bs = self.bs if bs is None else bs
        theta = torch.zeros([bs, 2, 3], dtype=self.dtype, device=self.device)
        theta[:, 0, 0] += cos ; theta[:, 0, 1] += -sin ; theta[:, 0, 2] += - x * cos + y * sin
        theta[:, 1, 0] += sin ; theta[:, 1, 1] += cos ;  theta[:, 1, 2] += - x * sin - y * cos
        return theta

    def forward(self, X, u):
        [_, self.bs, c, self.d, self.d] = X.shape
        T = len(self.t_eval)
        # encode
        self.r0_m, self.r0_v, self.phi0_m, self.phi0_v, self.phi0_m_n = self.encode(X[0])
        self.r1_m, self.r1_v, self.phi1_m, self.phi1_v, self.phi1_m_n = self.encode(X[1])

        # reparametrize
        self.Q_r0 = Normal(self.r0_m, self.r0_v)
        self.P_normal = Normal(torch.zeros_like(self.r0_m), torch.ones_like(self.r0_v))
        self.r0 = self.Q_r0.rsample()

        self.Q_phi0 = VonMisesFisher(self.phi0_m_n, self.phi0_v)
        self.P_hyper_uni = HypersphericalUniform(1, device=self.device)
        self.phi0 = self.Q_phi0.rsample()
        while torch.isnan(self.phi0).any():
            self.phi0 = self.Q_phi0.rsample()

        # estimate velocity
        self.r_dot0 = (self.r1_m - self.r0_m) / (self.t_eval[1] - self.t_eval[0])
        self.phi_dot0 = self.angle_vel_est(self.phi0_m_n, self.phi1_m_n, self.t_eval[1]-self.t_eval[0])

        # predict
        z0_u = torch.cat([self.r0, self.phi0, self.r_dot0, self.phi_dot0, u], dim=1)
        zT_u = odeint(self.ode, z0_u, self.t_eval, method=self.hparams.solver) # T, bs, 4
        self.qT, self.q_dotT, _ = zT_u.split([3, 2, 2], dim=-1)
        self.qT = self.qT.view(T*self.bs, 3)

        # decode
        ones = torch.ones_like(self.qT[:, 0:1])
        self.cart = self.obs_net_1(ones)
        self.pole = self.obs_net_2(ones)

        theta1 = self.get_theta_inv(1, 0, self.qT[:, 0], 0, bs=T*self.bs)
        theta2 = self.get_theta_inv(self.qT[:, 1], self.qT[:, 2], self.qT[:, 0], 0, bs=T*self.bs)

        grid1 = F.affine_grid(theta1, torch.Size((T*self.bs, 1, self.d, self.d)))
        grid2 = F.affine_grid(theta2, torch.Size((T*self.bs, 1, self.d, self.d)))

        transf_cart = F.grid_sample(self.cart.view(T*self.bs, 1, self.d, self.d), grid1)
        transf_pole = F.grid_sample(self.pole.view(T*self.bs, 1, self.d, self.d), grid2)
        self.Xrec = torch.cat([transf_cart, transf_pole, torch.zeros_like(transf_cart)], dim=1)
        self.Xrec = self.Xrec.view(T, self.bs, 3, self.d, self.d)
        return None

    def training_step(self, train_batch, batch_idx):
        X, u = train_batch
        self.forward(X, u)

        lhood = - self.loss_fn(self.Xrec, X)
        lhood = lhood.sum([0, 2, 3, 4]).mean()
        kl_q = torch.distributions.kl.kl_divergence(self.Q_r0, self.P_normal).mean() + \
                torch.distributions.kl.kl_divergence(self.Q_phi0, self.P_hyper_uni).mean()
        norm_penalty = (self.phi0_m.norm(dim=-1).mean() - 1) ** 2

        loss = - lhood + kl_q + 1/100 * norm_penalty

        logs = {'recon_loss': -lhood, 'kl_q_loss': kl_q, 'train_loss': loss}
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
        parser.add_argument('--learning_rate', default=1e-4, type=float)
        parser.add_argument('--batch_size', default=1024, type=int)

        return parser


def main(args):
    model = Model(hparams=args, data_path=os.path.join(PARENT_DIR, 'datasets', 'cartpole-gym-image-dataset-rgb-u9-train.pkl'))

    checkpoint_callback = ModelCheckpoint(monitor='loss', 
                                          prefix=args.name+f'-T_p={args.T_pred}-', 
                                          save_top_k=1, 
                                          save_last=True)
    trainer = Trainer.from_argparse_args(args, 
                                         deterministic=True,
                                         default_root_dir=os.path.join(PARENT_DIR, 'logs', args.name),
                                         checkpoint_callback=checkpoint_callback) 
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--name', default='ablation-cart-MLPdyna-cavae', type=str)
    parser.add_argument('--T_pred', default=4, type=int)
    parser.add_argument('--solver', default='euler', type=str)
    # add args from trainer
    parser = Trainer.add_argparse_args(parser)
    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = Model.add_model_specific_args(parser)
    # parse params
    args = parser.parse_args()

    main(args)