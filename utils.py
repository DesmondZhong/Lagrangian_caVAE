import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

def to_pickle(thing, path): # save something
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)


def from_pickle(path): # load something
    thing = None
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing

def arrange_data(x, us, t, num_points=2):
    # assume x has shape 
    #    n_u, ts, bs, 3
    # or n_u, ts, bs, 32, 32
    # or n_u, ts, bs, 3, 64, 64 
    # output x has shape n_u, num_points, bs * (ts-num_points+1), ... 
    assert num_points>=2 and num_points<=len(t)
    n_u, ts, bs = x.shape[0:3]
    x_list = []
    u_list = []
    for u_ind in range(n_u):
        temp = np.zeros((num_points, bs*(ts-num_points+1), *x.shape[3:]), dtype=np.float32)
        for i in range(ts-num_points+1):
            temp[:, i*bs:(i+1)*bs] = x[u_ind, i:i+num_points] # n_u, num_points, bs, ...
        x_list.append(temp)
        u_array = np.array(us[u_ind:u_ind+1], dtype=np.float32)
        u_list.append(u_array * np.ones((temp.shape[1], len(u_array)), dtype=np.float32))
    t_eval=t[0:num_points]
    return np.concatenate(x_list, axis=1), np.concatenate(u_list, axis=0), t_eval

def my_collate(batch):
    r"""collate a batch so that batch size not in the first dimension for x
        but it is the first dimension for u
    """

    x, u = zip(*batch)
    # turn into tensor
    x = [torch.as_tensor(b) for b in x]
    u = [torch.as_tensor(b) for b in u]
    # collate x
    elem = x[0]
    out = None
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum([b.numel() for b in x])
        storage = elem.storage()._new_shared(numel)
        out = elem.new(storage)
    x_collate = torch.stack(x, 1, out=out)
    # collate u
    elem = u[0]
    out = None
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum([b.numel() for b in u])
        storage = elem.storage()._new_shared(numel)
        out = elem.new(storage)
    u_collate = torch.stack(u, 0, out=out)

    return [x_collate, u_collate]

class ImageDataset(Dataset):
    def __init__(self, data_path, T_pred, ctrl=True):
        data = from_pickle(data_path)
        if ctrl:
            self.x, self.u, self.t_eval = arrange_data(data['x'], data['us'], data['t'], num_points=T_pred+1)
            self.obs, _, _ = arrange_data(data['obs'], data['us'], data['t'], num_points=T_pred+1)
        else:
            self.x, self.u, self.t_eval = arrange_data(data['x'][0:1], data['us'][0:1], data['t'], num_points=T_pred+1)
            self.obs, _, _ = arrange_data(data['obs'][0:1], data['us'][0:1], data['t'], num_points=T_pred+1)

    def __getitem__(self, index):
        return (self.x[:, index], self.u[index])

    def __len__(self):
        return self.u.shape[0]


class HomoImageDataset(Dataset):
    def __init__(self, data_path, T_pred):
        data = from_pickle(data_path)
        self.x = [] ; self.u = [] ; self.obs = []
        for i in range(data['x'].shape[0]):
            x, u, self.t_eval = arrange_data(data['x'][i:i+1], data['us'][i:i+1], data['t'], num_points=T_pred+1)
            obs, _, _ = arrange_data(data['obs'][i:i+1], data['us'][i:i+1], data['t'], num_points=T_pred+1)
            self.x.append(x) ; self.u.append(u) ; self.obs.append(obs)
        self.u_idx = 0

    def __getitem__(self, index):
        return (self.x[self.u_idx][:, index], self.u[self.u_idx][index])

    def __len__(self):
        return self.u[self.u_idx].shape[0]