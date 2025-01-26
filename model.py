import torch as th
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Optional
from torch_scatter import scatter_mean

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')


class hard_fc(th.nn.Module):
    def __init__(self, d_in, d_hid, DroPout=0):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(DroPout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)

        return x


class GCN_Net(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats, out_feats)
        self.fc = th.nn.Linear(2 * out_feats, 4)
        self.hard_fc1 = hard_fc(out_feats, out_feats)

    def forward(self, x, edge_index, batch):
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        x1 = self.conv2(x1, edge_index)
        x1 = F.relu(x1)
        x1 = scatter_mean(x1, batch, dim=0)
        x1_g = x1
        x1 = self.hard_fc1(x1)
        x1_t = x1
        x1 = th.cat((x1_g, x1_t), 1)

        return x1


class RERG(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(RERG, self).__init__()
        self.GCN_Net = GCN_Net(in_feats, hid_feats, out_feats)
        self.fc = th.nn.Linear(2 * out_feats, 4)

    def forward(self, data):
        z1 = self.GCN_Net(data.x0, data.edge_index1, data.batch)
        z2 = self.GCN_Net(data.x, data.edge_index2, data.batch)
        z_combin = th.cat((z1, z2), 0)
        z_combin = self.fc(z_combin)
        y_pre = F.log_softmax(z_combin, dim=1)

        return z1, z2, y_pre

    def projection(self, z: th.Tensor) -> th.Tensor:
        fc1 = nn.Linear(64, 64).to(device)
        fc2 = nn.Linear(64, 64).to(device)
        z = F.elu(fc1(z))
        return fc2(z)

    def sim(self, z1: th.Tensor, z2: th.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return th.mm(z1, z2.t())

    def semi_loss(self, z1: th.Tensor, z2: th.Tensor):
        f = lambda x: th.exp(x / 0.5)
        refl_sim = self.sim(z1, z1)
        between_sim = self.sim(z1, z2)
        refl_sim = f(refl_sim)
        between_sim = f(between_sim)
        return -th.log(between_sim.diag() / (between_sim.sum(1) + refl_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: th.Tensor, z2: th.Tensor,
                          batch_size: int):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: th.exp(x / self.tau)
        indices = th.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-th.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return th.cat(losses)

    def loss(self, z1: th.Tensor, z2: th.Tensor, epoch, epoch_start, mean: bool = True,
             batch_size: Optional[int] = None):

        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if epoch < epoch_start:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret
