import torch
import torch.nn as nn
import numpy as np
# from utils import choose_nonlinearity
from torch.nn import functional as F

def choose_nonlinearity(name):
    nl = None
    if name == 'tanh':
        nl = torch.tanh
    elif name == 'relu':
        nl = torch.relu
    elif name == 'sigmoid':
        nl = torch.sigmoid
    elif name == 'softplus':
        nl = torch.nn.functional.softplus
    elif name == 'selu':
        nl = torch.nn.functional.selu
    elif name == 'elu':
        nl = torch.nn.functional.elu
    elif name == 'swish':
        nl = lambda x: x * torch.sigmoid(x)
    else:
        raise ValueError("nonlinearity not recognized")
    return nl

class MLP(torch.nn.Module):
    '''Just a MLP'''
    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh', bias_bool=True):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=bias_bool)

        for l in [self.linear1, self.linear2, self.linear3]:
            torch.nn.init.orthogonal_(l.weight) # use a principled initialization

        self.nonlinearity = choose_nonlinearity(nonlinearity)

    def forward(self, x, separate_fields=False):
        h = self.nonlinearity( self.linear1(x) )
        h = self.nonlinearity( self.linear2(h) )
        return self.linear3(h)


class PSD(torch.nn.Module):
    '''A Neural Net which outputs a positive semi-definite matrix'''
    def __init__(self, input_dim, hidden_dim, diag_dim, nonlinearity='tanh'):
        super(PSD, self).__init__()
        self.diag_dim = diag_dim
        if diag_dim == 1:
            self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
            self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.linear3 = torch.nn.Linear(hidden_dim, diag_dim)

            for l in [self.linear1, self.linear2, self.linear3]:
                torch.nn.init.orthogonal_(l.weight) # use a principled initialization

            self.nonlinearity = choose_nonlinearity(nonlinearity)
        else:
            assert diag_dim > 1
            self.diag_dim = diag_dim
            self.off_diag_dim = int(diag_dim * (diag_dim - 1) / 2)
            self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
            self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.linear4 = torch.nn.Linear(hidden_dim, self.diag_dim + self.off_diag_dim)

            for l in [self.linear1, self.linear2, self.linear3, self.linear4]:
                torch.nn.init.orthogonal_(l.weight) # use a principled initialization

            self.nonlinearity = choose_nonlinearity(nonlinearity)

    def forward(self, q):
        if self.diag_dim == 1:
            h = self.nonlinearity( self.linear1(q) )
            h = self.nonlinearity( self.linear2(h) )
            h = self.nonlinearity( self.linear3(h) )
            return h*h + 0.1
        else:
            bs = q.shape[0]
            h = self.nonlinearity( self.linear1(q) )
            h = self.nonlinearity( self.linear2(h) )
            h = self.nonlinearity( self.linear3(h) )
            diag, off_diag = torch.split(self.linear4(h), [self.diag_dim, self.off_diag_dim], dim=1)
            # diag = torch.nn.functional.relu( self.linear4(h) )

            L = torch.diag_embed(diag)

            ind = np.tril_indices(self.diag_dim, k=-1)
            flat_ind = np.ravel_multi_index(ind, (self.diag_dim, self.diag_dim))
            L = torch.flatten(L, start_dim=1)
            L[:, flat_ind] = off_diag
            L = torch.reshape(L, (bs, self.diag_dim, self.diag_dim))

            D = torch.bmm(L, L.permute(0, 2, 1))
            for i in range(self.diag_dim):
                D[:, i, i] += 0.1
            # D[:, 0, 0] = D[:, 0, 0] + 0.1
            # D[:, 1, 1] = D[:, 1, 1] + 0.1
            return D


class MLP_Encoder(torch.nn.Module):
    '''Just a MLP Encoder with residual term'''
    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh', bias_bool=True):
        super(MLP_Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        # self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, output_dim, bias=bias_bool)

        # for l in [self.linear1, self.linear2, self.linear3, self.linear4]:
        #     torch.nn.init.orthogonal_(l.weight) # use a principled initialization

        self.nonlinearity = choose_nonlinearity(nonlinearity)

    def forward(self, x, separate_fields=False):
        h = self.nonlinearity( self.linear1(x) )
        h = self.nonlinearity( self.linear2(h) )
        # h = self.nonlinearity( self.linear3(h) )
        return self.linear4(h)

class MLP_Decoder(torch.nn.Module):
    '''MLP Decoder'''
    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh', bias_bool=True):
        super(MLP_Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        # self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, output_dim, bias=bias_bool)
        self.sigmoid = torch.nn.Sigmoid()

        # for l in [self.linear1, self.linear2, self.linear3, self.linear4]:
        #     torch.nn.init.orthogonal_(l.weight) # use a principled initialization

        self.nonlinearity = choose_nonlinearity(nonlinearity)

    def forward(self, x, separate_fields=False):
        h = self.nonlinearity( self.linear1(x) )
        # h = self.nonlinearity( self.linear2(h) )
        h = self.nonlinearity( self.linear3(h) )
        return self.linear4(h)


class Encoder(torch.nn.Module):
    '''CNN Encoder'''
    def __init__(self, in_channels, out_dim, nonlinearity='relu'):
        super(Encoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 32, 3, stride=2)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=2)
        self.fc1 = torch.nn.Linear(32*7*7, 10)
        self.fc2 = torch.nn.Linear(10, out_dim)

        for l in [self.conv1, self.conv2, self.fc1, self.fc2]:
            torch.nn.init.orthogonal_(l.weight) # use a principled initialization

        self.nonlinearity = choose_nonlinearity(nonlinearity)

    def forward(self, x):
        h = self.nonlinearity(self.conv1(x))
        h = self.nonlinearity(self.conv2(h))
        h = self.nonlinearity(self.fc1(h.view(-1, 32*7*7)))
        return self.fc2(h)

class Decoder(torch.nn.Module):
    '''CNN Decoder'''
    def __init__(self, in_dim, out_channels, nonlinearity='relu'):
        super(Decoder, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, 10)
        self.fc2 = torch.nn.Linear(10, 32*7*7)
        self.conv1_T = torch.nn.ConvTranspose2d(32, 32, 3, stride=2)
        self.conv2_T = torch.nn.ConvTranspose2d(32, out_channels, 3, stride=2, output_padding=1)
        self.sigmoid = torch.nn.Sigmoid()

        for l in [self.conv1_T, self.conv2_T, self.fc1, self.fc2]:
            torch.nn.init.orthogonal_(l.weight) # use a principled initialization

        self.nonlinearity = choose_nonlinearity(nonlinearity)

    def forward(self, x):
        h = self.nonlinearity(self.fc1(x))
        h = self.nonlinearity(self.fc2(h))
        h = self.nonlinearity(self.conv1_T(h.view(-1, 32, 7, 7)))
        return self.sigmoid(self.conv2_T(h))

class MLP_Wrapper(torch.nn.Module):
    ''' a MLP wrapper that output a zero in the last component'''
    def __init__(self, state_u_dim, hidden, state_dim, nonlinearity='relu'):
        super(MLP_Wrapper, self).__init__()
        self.state_dim = state_dim
        self.u_dim = state_u_dim - state_dim
        self.mlp = MLP(state_u_dim, hidden, state_dim, nonlinearity=nonlinearity)

    def forward(self, t, x):
        h = self.mlp(x)
        return torch.cat((h, torch.zeros_like(h[:,0:self.u_dim])), dim=1)


class MLP_prob_Wrapper(torch.nn.Module):
    ''' a MLP wrapper that also deal with probability dynamics'''
    def __init__(self, hidden_dim, state_dim, u_dim, log_prob_dim, nonlinearity='relu'):
        super(MLP_prob_Wrapper, self).__init__()
        self.state_dim = state_dim
        self.u_dim = u_dim
        self.log_prob_dim = log_prob_dim
        self.mlp = MLP(state_dim, hidden_dim, state_dim, nonlinearity=nonlinearity)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        # with torch.enable_grad():
        # one = torch.ones_like(x[0][0, 0]) ; one.requires_grad = True
        z, z_logp = x
        # state = state * one
        # state_u = torch.cat((state, u), dim=1)
        dzdt = self.mlp(z)
        # Jacobian = [torch.autograd.grad(state_f[:, i].sum(), state, create_graph=True)[0] for i in range(self.state_dim)] # every element (bs, state_dim)
        Jacobian = []
        for i in range(self.state_dim):
            row = torch.autograd.grad(dzdt[:, i].sum(), z, create_graph=True)[0]
            Jacobian.append(row)
        dz_logp_dt = -sum([Jacobian[i][:, i] for i in range(self.state_dim)]).view(-1, 1)
        # dlog_prob_q = -sum([Jacobian[i][:, i] for i in range(self.state_dim//2)]).view(-1, 1)
        # dlog_prob_p = -sum([Jacobian[i][:, i] for i in range(self.state_dim//2, self.state_dim)]).view(-1, 1)

        # df_ds = torch.stack(
        #     [torch.autograd.grad(state_f[:, i], state, torch.ones_like(state_f[:, i]),
        #     retain_graph=True, create_graph=True)[0].contiguous()[:, i]
        #     for i in range(self.state_dim)], 1
        # )
        # df_dp = torch.stack(
        #     [torch.autograd.grad(state_f_p[:, i], state, torch.ones_like(state_f_p[:, i]),
        #     retain_graph=True, create_graph=True)[0].contiguous()[:, i]
        #     for i in range(self.state_dim//2)], 1
        # )
        # dlog_prob_z = -torch.sum(df_dp, 1).view(-1, 1)
        return (dzdt, dz_logp_dt)


class Geometric_Baseline(torch.nn.Module):
    ''' Geometric_Baseline of one dimensional'''
    def __init__(self, nonlinearity='tanh'):
        super(Geometric_Baseline, self).__init__()
        self.mlp = MLP(3, 600, 1, nonlinearity=nonlinearity)
        self.g = MLP(3, 100, 1, nonlinearity=nonlinearity)

    def forward(self, t, x):
        z, u = x.split([3, 1], dim=1)
        cos_q, sin_q, q_dot = z.split([1, 1, 1], dim=1)

        d_cos_q = - sin_q * q_dot
        d_sin_q = cos_q * q_dot
        d_q_dot = self.mlp(z) + self.g(z) * u
        return torch.cat([d_cos_q, d_sin_q, d_q_dot, torch.zeros_like(u)], dim=1)


class MatrixNet(torch.nn.Module):
    ''' a neural net which outputs a matrix'''
    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh', bias_bool=True, shape=(2,2)):
        super(MatrixNet, self).__init__()
        self.mlp = MLP(input_dim, hidden_dim, output_dim, nonlinearity, bias_bool)
        self.shape = shape

    def forward(self, x):
        flatten = self.mlp(x)
        return flatten.view(-1, *self.shape)


class U_Net(torch.nn.Module):
    def __init__(self, base_channels, out_channels, nonlinearity='relu'):
        super(U_Net, self).__init__()
        bc = base_channels ; oc = out_channels
        self.nonlinearity = choose_nonlinearity(nonlinearity) 
        self.conv1_1 = nn.Conv2d(1, bc, 3, padding=1)
        self.conv1_2 = nn.Conv2d(bc, bc, 3, padding=1)
        self.maxPool = nn.MaxPool2d(2)
        self.conv2_1 = nn.Conv2d(bc, 2*bc, 3, padding=1)
        self.conv2_2 = nn.Conv2d(2*bc, 2*bc, 3, padding=1)
        self.conv3_1 = nn.Conv2d(2*bc, 4*bc, 3, padding=1)
        self.conv3_2 = nn.Conv2d(4*bc, 4*bc, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_upsample1 = nn.Conv2d(4*bc, 2*bc, 3, padding=1)
        self.conv4_1 = nn.Conv2d(4*bc, 2*bc, 3, padding=1)
        self.conv4_2 = nn.Conv2d(2*bc, 2*bc, 3, padding=1)
        self.conv_upsample2 = nn.Conv2d(2*bc, bc, 3, padding=1)
        self.conv5_1 = nn.Conv2d(2*bc, bc, 3, padding=1)
        self.conv5_2 = nn.Conv2d(bc, bc, 3, padding=1)
        self.out_conv = nn.Conv2d(bc, oc, 1)


    def forward(self, input_img):
        # contracting path
        h = self.nonlinearity(self.conv1_1(input_img)) # bs * bc * 32 * 32
        h1 = self.nonlinearity(self.conv1_2(h)) # bs * bc * 32 * 32 
        h = self.maxPool(h1) # bs * bc * 16 * 16 
        h = self.nonlinearity(self.conv2_1(h)) # bs * 2*bc * 16 * 16 
        h2 = self.nonlinearity(self.conv2_2(h)) # bs * 2*bc * 16 * 16 
        h = self.maxPool(h2) # bs * 2*bc * 8 * 8 
        h = self.nonlinearity(self.conv3_1(h)) # bs * 4*bc * 8 * 8 
        h = self.nonlinearity(self.conv3_2(h)) # bs * 4*bc * 8 * 8 
        # expansion path
        h = self.upsample(h) # bs * 4*bc * 16 * 16
        h = self.conv_upsample1(h) # bs * 2*bc * 16 * 16
        h = torch.cat([h, h2], dim=1) # bs * 4*bc * 16 * 16
        h = self.nonlinearity(self.conv4_1(h)) # bs * 2*bc * 16 * 16
        h = self.nonlinearity(self.conv4_2(h)) # bs * 2*bc * 16 * 16
        h = self.upsample(h) # bs * 2*bc * 32 * 32
        h = self.conv_upsample2(h) # bs * bc * 32 * 32
        h = torch.cat([h, h1], dim=1) # bs * 2*bc * 32 * 32
        h = self.nonlinearity(self.conv5_1(h)) # bs * bc * 32 * 32
        h = self.nonlinearity(self.conv5_2(h)) # bs * bc * 32 * 32

        h = self.out_conv(h) # bs * oc * 32 * 32

        return h 


class Conv_Encoder(torch.nn.Module):
    def __init__(self, d=64, output_dim=3, nonlinearity='elu'):
        super(Conv_Encoder, self).__init__()
        self.nonlinearity = choose_nonlinearity(nonlinearity)
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1) 
        self.maxPool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 6, 3, padding=1)
        self.conv3 = nn.Conv2d(6, 6, 3, padding=1)
        self.conv4 = nn.Conv2d(6, 6, 3, padding=1)
        self.d = d
        if d == 64:
            self.fc = nn.Linear(96, output_dim)
        elif d == 32:
            self.fc = nn.Linear(24, output_dim)
        else:
            raise NotImplementedError

    def forward(self, input_img):
        # bs, 1, 64, 64
        h = self.nonlinearity(self.conv1(input_img)) # 6, 64, 64
        h = self.maxPool(h) # 6, 32, 32
        h = self.nonlinearity(self.conv2(h)) # 6, 32, 32
        h = self.maxPool(h) # 6, 16, 16
        h = self.nonlinearity(self.conv3(h)) # 6, 16, 16
        h = self.maxPool(h) # 6, 8, 8
        h = self.nonlinearity(self.conv4(h)) # 6, 8, 8
        h = self.maxPool(h) # 6, 4, 4
        if self.d == 64:
            h = self.fc(h.view(-1, 96)) # bs, output_dim
        elif self.d == 32:
            h = self.fc(h.view(-1, 24)) # bs, output_dim
        else:
            raise NotImplementedError
        return h



class Res_Block(torch.nn.Module):
    def __init__(self):
        super(Res_Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU()
        )

    def forward(x):
        h = self.block(x)
        h += x
        return F.sigmoid(h)


class HGN_Decoder(torch.nn.Module):
    def __init__(self, out_c):
        super(HGN_Decoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2d_0 = nn.Conv2d(16, 64, 1)
        self.res_block_1 = Res_Block()
        self.res_block_2 = Res_Block()
        self.res_block_3 = Res_Block()
        self.conv2d_end = nn.Conv2d(64, out_c, 1)

    def forward(self, q):
        # assume q has dimension 4, 4, 16
        h = self.conv2d_0(self.upsample(q)) # 4,4,16 -> 8, 8, 16 -> 8,8,64
        h = self.upsample(self.res_block_1(h)) # 16, 16, 64
        h = self.upsample(self.res_block_2(h)) # 32, 32, 64
        h = self.res_block_3(h)
        return F.sigmoid()