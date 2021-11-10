import torch
import numpy as np


class Lag_Net(torch.nn.Module):

    def __init__(self, q_dim=1, u_dim=1, 
                g_net=None, M_net=None, V_net=None,g_baseline=None, dyna_model='lag'):
        super(Lag_Net, self).__init__()
        self.dyna_model = dyna_model
        self.q_dim = q_dim
        self.u_dim = u_dim
        # neural networks
        self.g_net = g_net
        if self.dyna_model == 'lag':
            self.M_net = M_net
            self.V_net = V_net
        elif self.dyna_model == 'g_baseline':
            self.g_baseline = g_baseline
        else:
            NotImplementedError

    def forward(self, t, x):
        if self.dyna_model == 'g_baseline':
            cos_q_sin_q_q_dot, u = x.split([3*self.q_dim, self.u_dim], dim=1)
            cos_q_sin_q, q_dot = cos_q_sin_q_q_dot.split([2*self.q_dim, self.q_dim], dim=1)
            cos_q, sin_q = cos_q_sin_q.split([self.q_dim, self.q_dim], dim=1)
            d_cos_q = - sin_q * q_dot
            d_sin_q = cos_q * q_dot
            if self.q_dim == 1:
                d_q_dot = self.g_baseline(cos_q_sin_q_q_dot) + self.g_net(cos_q_sin_q) * u
            else:
                d_q_dot = self.g_baseline(cos_q_sin_q_q_dot)
                d_q_dot += torch.squeeze(torch.matmul(self.g_net(cos_q_sin_q), torch.unsqueeze(u, dim=2)), dim=2)
            return torch.cat([d_cos_q, d_sin_q, d_q_dot, torch.zeros_like(u)], dim=1)

        cos_q, sin_q, q_dot, u = x.split([self.q_dim, self.q_dim, self.q_dim, self.u_dim], dim=1)
        cos_q_sin_q = torch.cat((cos_q, sin_q), dim=1)
        d_cos_q = - sin_q * q_dot
        d_sin_q = cos_q * q_dot

        # d_q_dot is where lagrangian plays a role
        self.M_q = self.M_net(cos_q_sin_q)
        self.V_q = self.V_net(cos_q_sin_q)
        dV = torch.autograd.grad(self.V_q.sum(), cos_q_sin_q, create_graph=True)[0] # bs, 2*self.q_dim
        dV_dq = dV[:,0:self.q_dim] * (- sin_q) + dV[:, self.q_dim:2*self.q_dim] * cos_q # bs, self.q_dim
        if self.q_dim == 1:
            dM = torch.autograd.grad(self.M_q.sum(), cos_q_sin_q, create_graph=True)[0] # bs, 2*self.q_dim
            dM_dq = dM[:,0:self.q_dim] * (- sin_q) + dM[:, self.q_dim:2*self.q_dim] * cos_q # bs, self.q_dim
            d_q_dot = (-0.5 * q_dot * q_dot * dM_dq - dV_dq + self.g_net(cos_q_sin_q) * u)
            d_q_dot = d_q_dot / self.M_q
        else:
            dM_dt = torch.zeros_like(self.M_q)
            for row_ind in range(self.q_dim):
                for col_ind in range(self.q_dim):
                    dM = torch.autograd.grad(self.M_q[:, row_ind, col_ind].sum(), cos_q_sin_q, create_graph=True)[0]
                    dM_dt[:, row_ind, col_ind] = (dM * torch.cat((-sin_q * q_dot, cos_q * q_dot), dim=1)).sum(-1)
            q_dot_M_q_dot = torch.matmul(
                torch.unsqueeze(q_dot, 1),
                torch.matmul(self.M_q, torch.unsqueeze(q_dot, 2))
            ) # (bs, 1, 1)
            d_q_dot_M_q_dot = torch.autograd.grad(q_dot_M_q_dot.sum(), cos_q_sin_q, create_graph=True)[0]
            d_q_dot_M_q_dot_dq = d_q_dot_M_q_dot[:, 0:self.q_dim] * (- sin_q) \
                                + d_q_dot_M_q_dot[:, self.q_dim:2*self.q_dim] * cos_q
            temp = - torch.matmul(dM_dt, q_dot[:, :, None]) \
                    + 0.5 * d_q_dot_M_q_dot_dq[:, :, None] \
                    - dV_dq[:, :, None] \
                    + torch.matmul(self.g_net(cos_q_sin_q), u[:, :, None])
            d_q_dot = torch.squeeze(torch.matmul(torch.inverse(self.M_q), temp), 2)

        return torch.cat([d_cos_q, d_sin_q, d_q_dot, torch.zeros_like(u)], dim=1)

class Lag_Net_R1_T1(torch.nn.Module):
    def __init__(self, g_net=None, M_net=None, V_net= None, g_baseline=None, dyna_model='lag'):
        super(Lag_Net_R1_T1, self).__init__()
        self.dyna_model = dyna_model
        self.g_net = g_net
        if self.dyna_model == 'lag':
            self.M_net = M_net
            self.V_net = V_net
        elif self.dyna_model == 'g_baseline':
            self.g_baseline = g_baseline
        else:
            raise NotImplementedError

    def forward(self, t, x):
        if self.dyna_model == 'g_baseline':
            r_cos_phi_sin_phi_r_dot_phi_dot, u = x.split([5, 2], dim=1)
            r_cos_phi_sin_phi, r_dot_phi_dot = r_cos_phi_sin_phi_r_dot_phi_dot.split([3, 2], dim=1)
            r, cos_phi, sin_phi = r_cos_phi_sin_phi.split([1, 1, 1], dim=1)
            r_dot, phi_dot = r_dot_phi_dot.split([1, 1], dim=1)
            dcos_phi = - sin_phi * phi_dot
            dsin_phi = cos_phi * phi_dot
            dr = r_dot

            dq_dot = self.g_baseline(r_cos_phi_sin_phi_r_dot_phi_dot)
            dq_dot += torch.squeeze(torch.matmul(self.g_net(r_cos_phi_sin_phi), torch.unsqueeze(u, dim=2)), dim=2)
            return torch.cat([dr, dcos_phi, dsin_phi, dq_dot, torch.zeros_like(u)], dim=1)


        r, cos_phi, sin_phi, r_dot, phi_dot, u = x.split([1, 1, 1, 1, 1, 2], dim=1)
        dcos_phi = - sin_phi * phi_dot
        dsin_phi = cos_phi * phi_dot
        dr = r_dot

        r_cos_phi_sin_phi = torch.cat([r, cos_phi, sin_phi], dim=1)

        # d_r_dot d_q_dot is where Lagrangian plays a role
        self.M_q = self.M_net(r_cos_phi_sin_phi)
        self.V_q = self.V_net(r_cos_phi_sin_phi)
        dV = torch.autograd.grad(self.V_q.sum(), r_cos_phi_sin_phi, create_graph=True)[0]
        dV_dr, dV_dcos_phi, dV_dsin_phi = dV.split([1, 1, 1], dim=1)
        dV_dphi = dV_dcos_phi * (- sin_phi) + dV_dsin_phi * cos_phi
        dM_dt = torch.zeros_like(self.M_q)
        for row_ind in range(2):
            for col_ind in range(2):
                dM = torch.autograd.grad(self.M_q[:, row_ind, col_ind].sum(), r_cos_phi_sin_phi, create_graph=True)[0]
                dM_dt[:, row_ind, col_ind] = (dM * torch.cat([r_dot, -sin_phi*phi_dot, cos_phi*phi_dot], dim=1)).sum(-1)
        q_dot = torch.cat([r_dot, phi_dot], dim=1)
        q_dot_M_q_dot = torch.matmul(
            q_dot[:, None, :],
            torch.matmul(self.M_q, q_dot[:, :, None])
        ) # (bs, 1, 1)
        d_q_dot_M_q_dot = torch.autograd.grad(q_dot_M_q_dot.sum(), r_cos_phi_sin_phi, create_graph=True)[0]
        d_q_dot_M_q_dot_dr, d_q_dot_M_q_dot_dcos_phi, d_q_dot_M_q_dot_dsin_phi = \
            d_q_dot_M_q_dot.split([1, 1, 1], dim=1)
        d_q_dot_M_q_dot_dphi = d_q_dot_M_q_dot_dcos_phi * (- sin_phi) + \
                                d_q_dot_M_q_dot_dsin_phi * cos_phi
        d_q_dot_M_q_dot_dq = torch.cat([d_q_dot_M_q_dot_dr, d_q_dot_M_q_dot_dphi], dim=1)
        temp = - torch.matmul(dM_dt, q_dot[:, :, None]) \
                + 0.5 * d_q_dot_M_q_dot_dq[:, :, None] \
                - torch.cat([dV_dr, dV_dphi], dim=1)[:, :, None]\
                + torch.matmul(self.g_net(r_cos_phi_sin_phi), u[:, :, None])
        dq_dot  = torch.squeeze(torch.matmul(torch.inverse(self.M_q), temp), dim=2)
        return torch.cat([dr, dcos_phi, dsin_phi, dq_dot, torch.zeros_like(u)], dim=1)
