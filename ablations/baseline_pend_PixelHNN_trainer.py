
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

# local application imports
from ablations.PixelHNN import HNN, PixelHNN, MLP, MLPAutoencoder, PixelHNNDataset
from utils import arrange_data, from_pickle, my_collate


seed_everything(0)


class Model(pl.LightningModule):

    def __init__(self, hparams, data_path=None):
        super(Model, self).__init__()
        self.hparams = hparams
        self.data_path = data_path
        self.loss_fn = torch.nn.MSELoss(reduction='none')

        autoencoder = MLPAutoencoder(input_dim=2*32**2, hidden_dim=200, latent_dim=2, nonlinearity='relu')
        self.pixelHNN = PixelHNN(input_dim=2, hidden_dim=200, autoencoder=autoencoder,
                                 nonlinearity='tanh')

    def train_dataloader(self):
        train_dataset = PixelHNNDataset(self.data_path)
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True)

    def training_step(self, train_batch, batch_idx):
        x, x_next = train_batch

        z = self.pixelHNN.encode(x)
        z_next = self.pixelHNN.encode(x_next)

        x_hat = self.pixelHNN.decode(z)

        # autoencoder loss
        ae_loss = ((x-x_hat)**2).mean()

        # hnn vector field loss
        z_hat_next = z + self.pixelHNN.time_derivative(z)
        hnn_loss = ((z_next - z_hat_next) **2).mean()

        # canonical coordinate loss
        # -> makes latent space look like (x, v) coordinates
        w, dw = z.split(1,1)
        w_next, _ = z_next.split(1,1)
        cc_loss = ((dw-(w_next - w))**2).mean()

        loss = ae_loss + cc_loss + 1e-1 * hnn_loss

        self.x_hat = x_hat
        self.z = z

        logs = {'train_loss': loss, 'ae_loss': ae_loss, 'cc_loss': cc_loss, 'hnn_loss': hnn_loss}
        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', default=1e-3, type=float)
        parser.add_argument('--batch_size', default=512, type=int)

        return parser


def main(args):
    model = Model(hparams=args, data_path=os.path.join(PARENT_DIR, 'datasets', 'pendulum-gym-image-dataset-train.pkl'))
    checkpoint_callback = ModelCheckpoint(monitor='loss', 
                                          prefix=args.name, 
                                          save_top_k=1, 
                                          save_last=True)
    trainer = Trainer.from_argparse_args(args, 
                                         deterministic=True,
                                         default_root_dir=os.path.join(PARENT_DIR, 'logs', args.name),
                                         checkpoint_callback=checkpoint_callback) 
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--name', default='baseline-pend-PixelHNN', type=str)
    # add args from trainer
    parser = Trainer.add_argparse_args(parser)
    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = Model.add_model_specific_args(parser)
    # parse params
    args = parser.parse_args()

    main(args)