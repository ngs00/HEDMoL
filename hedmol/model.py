import torch
import math
from torch.nn.functional import normalize, elu
from torch_scatter import scatter_add
from torch_geometric.typing import PairTensor
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing, EGConv, GINConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.glob import global_add_pool
from torch_geometric.utils import softmax


class Set2Set(torch.nn.Module):
    def __init__(self, in_channels, processing_steps, num_layers=1):
        super(Set2Set, self).__init__()
        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(self.out_channels, self.in_channels, num_layers)

        self.lstm.reset_parameters()

    def forward(self, x, batch):
        batch_size = batch.max().item() + 1
        h = (x.new_zeros((self.num_layers, batch_size, self.in_channels)),
             x.new_zeros((self.num_layers, batch_size, self.in_channels)))
        q_star = x.new_zeros(batch_size, self.out_channels)

        for i in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.in_channels)
            e = (x * q[batch]).sum(dim=-1, keepdim=True)
            a = softmax(e, batch, num_nodes=batch_size)
            r = scatter_add(a * x, batch, dim=0, dim_size=batch_size)
            q_star = torch.cat([q, r], dim=-1)

        return q_star


class EGC(torch.nn.Module):
    def __init__(self, n_node_feats, n_edge_feats, dim_state, dim_out):
        super(EGC, self).__init__()
        self.n_edge_feats = n_edge_feats
        self.dim_node_emb = 64
        self.dim_out = dim_out
        self.nfc = torch.nn.Linear(n_node_feats, 128)
        self.gc1 = EGConv(128, 128)
        self.gc2 = EGConv(128, self.dim_node_emb)
        self.attn = torch.nn.Linear(self.dim_node_emb + dim_state, 1)
        self.fc_g = torch.nn.Linear(self.dim_node_emb, dim_out)
        self.fc_cg = torch.nn.Linear(self.dim_node_emb, dim_out)

        self.nfc.reset_parameters()
        self.attn.reset_parameters()
        self.fc_g.reset_parameters()
        self.fc_cg.reset_parameters()

    def forward(self, g, z_e=None):
        h = elu(self.nfc(g.x))
        h = elu(self.gc1(h, g.edge_index))
        h = elu(self.gc2(h, g.edge_index))
        h = normalize(h, p=2, dim=1)

        # Calculate atom-level molecular embedding.
        z_a = global_add_pool(h, g.batch)
        z_a = self.fc_g(z_a)

        # Calculate conditional atom-level molecular embedding.
        h_elec_cond = torch.hstack([h, torch.repeat_interleave(z_e, g.n_atoms.squeeze(1), dim=0)])
        attns = softmax(self.attn(h_elec_cond) + 1e-16, g.batch)
        z_c = self.fc_cg(global_add_pool(attns * h, g.batch))

        return z_a, z_c, h


class GIN(torch.nn.Module):
    def __init__(self, n_node_feats, dim_out):
        super(GIN, self).__init__()
        self.dim_node_emb = 64
        self.dim_out = dim_out
        self.nfc = torch.nn.Linear(n_node_feats, 128)
        self.gc1 = GINConv(torch.nn.Linear(128, 128))
        self.gc2 = GINConv(torch.nn.Linear(128, self.dim_node_emb))
        self.set2set = Set2Set(self.dim_node_emb, processing_steps=4, num_layers=2)
        self.fc1 = torch.nn.Linear(2 * self.dim_node_emb, 32)
        self.fc2 = torch.nn.Linear(32, dim_out)
        self.act = torch.nn.GELU()

        self.nfc.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, g):
        h = self.act(self.nfc(g.x))
        h = self.act(self.gc1(h, g.edge_index))
        h = self.act(self.gc2(h, g.edge_index))
        h = normalize(h, p=2, dim=1)
        z_e = self.set2set(h, g.batch)
        z_e = self.act(self.fc1(z_e))
        z_e = self.fc2(z_e)

        return z_e, h


class PotentialEnergyConv(MessagePassing):
    def __init__(self, in_channels, out_channels, concat=True, beta=False,
                 dropout=0, edge_dim=None, bias=True, root_weight=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(PotentialEnergyConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], out_channels)
        self.lin_query = Linear(in_channels[1], out_channels)
        self.lin_value = Linear(in_channels[0], out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=None):
        C = self.out_channels

        if isinstance(x, torch.Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, 1, C)
        key = self.lin_key(x[0]).view(-1, 1, C)
        value = self.lin_value(x[0]).view(-1, 1, C)

        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, torch.Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i, key_j, value_j, edge_attr, index, ptr, size_i):
        if self.lin_edge is not None:
            assert edge_attr is not None

            key_j += self.lin_edge(edge_attr).view(-1, 1, self.out_channels)

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha

        return alpha.view(-1, 1, 1) * value_j


class HEDMOL(torch.nn.Module):
    def __init__(self, emb_net_a, emb_net_e, dim_out):
        super(HEDMOL, self).__init__()
        self.emb_net_a = emb_net_a
        self.emb_net_e = emb_net_e
        self.fc1 = torch.nn.Linear(2 * self.emb_net_a.dim_out, 64)
        self.fc2 = torch.nn.Linear(64, dim_out)
        self.eng_atom = PotentialEnergyConv(self.emb_net_a.dim_node_emb, self.emb_net_a.dim_node_emb)
        self.eng_net = torch.nn.Linear(self.emb_net_a.dim_node_emb, 1)

        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.eng_net.reset_parameters()

    def _predict(self, batch):
        h_jmg, ha_jmg = self.emb_net_e(batch.smg)
        h_mg, hc_mg, ha_mg = self.emb_net_a(batch.mg, normalize(h_jmg, p=2, dim=1))
        h = elu(self.fc1(torch.hstack([h_mg, hc_mg])))
        out = self.fc2(h)

        return out

    def forward(self, batch, predict=False):
        h_jmg, ha_jmg = self.emb_net_e(batch.smg)
        h_mg, hc_mg, ha_mg = self.emb_net_a(batch.mg, normalize(h_jmg, p=2, dim=1))
        h = elu(self.fc1(torch.hstack([h_mg, hc_mg])))
        out = self.fc2(h)

        if predict:
            return out
        else:
            if batch.substructs is None:
                eng_atoms = None
            else:
                h_suba = self.eng_atom(ha_mg[batch.idx_subatoms], batch.substructs.edge_index)
                h_suba = normalize(h_suba, p=2, dim=1)
                z_suba = global_add_pool(h_suba, batch.substructs.batch)
                eng_atoms = self.eng_net(z_suba)

            eng_substructs = self.eng_net(ha_jmg)

            return out, eng_atoms, eng_substructs

    def fit(self, data_loader, optimizer, criterion, alpha=0.2, coeff_pcl=1.0):
        train_loss = 0

        self.train()
        for batch in data_loader:
            batch.cuda()
            preds, eng_atoms, eng_substructs = self(batch)

            if coeff_pcl == 0:
                loss = criterion(preds, batch.y)
            else:
                loss = criterion(preds, batch.y)
                e_origin = batch.smg.energy.clone()
                energies_substructs = e_origin + torch.normal(mean=0, std=-0.1*e_origin).view(-1, 1)
                spcl = torch.abs(energies_substructs - eng_substructs) - alpha
                spcl[spcl < 0] = 0

                if eng_atoms is None:
                    loss += coeff_pcl * torch.mean(spcl)
                else:
                    energies_atoms = batch.substructs.eng + torch.normal(mean=0, std=-0.1*batch.substructs.eng).view(-1, 1)
                    apcl = torch.abs(energies_atoms - eng_atoms) - alpha
                    apcl[apcl < 0] = 0
                    loss += coeff_pcl * (torch.mean(apcl) + torch.mean(spcl))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        return train_loss / len(data_loader)

    def predict(self, data_loader):
        list_preds = list()

        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch.cuda()
                preds = self._predict(batch)
                list_preds.append(preds)

        return torch.vstack(list_preds).cpu()
