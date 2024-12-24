import numpy
import torch
from torch_geometric.utils.tree_decomposition import tree_decomposition
from torch_geometric.utils.convert import to_networkx
from karateclub.graph_embedding import GeoScattering
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


class AtomClique:
    def __init__(self, idx, atom_idx, feats, energy, substruct):
        self.idx = idx
        self.atom_idx = atom_idx
        self.feats = feats if isinstance(feats, torch.Tensor) else torch.tensor(feats, dtype=torch.float)
        self.energy = torch.tensor(energy, dtype=torch.float)
        self.substruct = substruct
        self.substruct.eng = self.energy


def assign_calc_attrs(dataset, dataset_calc, path_save_file=None):
    substructs = list()
    atom_clusters = list()
    edges_data = list()
    n_clqs_data = list()

    for n in range(0, len(dataset)):
        edges, idx_ac, n_clqs = tree_decomposition(dataset[n].mol)

        for i in range(0, n_clqs):
            idx_atoms = idx_ac[0][(idx_ac[1] == i).nonzero().view(-1)]
            substructs.append(dataset[n].mg.subgraph(idx_atoms))
            atom_clusters.append(idx_atoms)

        edges_data.append(edges)
        n_clqs_data.append(n_clqs)

    g = [to_networkx(d, to_undirected=True) for d in substructs]
    g += [to_networkx(d.mg, to_undirected=True) for d in dataset_calc]

    emb_model = GeoScattering()
    emb_model.fit(g)
    embs = emb_model.get_embedding()
    embs_g = embs[:len(substructs)]
    embs_s = embs[len(substructs):]

    pdists = pairwise_distances(embs_g, embs_s)
    nn_idx = numpy.argmin(pdists, axis=1)

    pos = 0
    for n in tqdm(range(0, len(dataset))):
        clqs = list()

        for i in range(0, n_clqs_data[n]):
            idx = pos + i
            mol_feats = dataset_calc[nn_idx[pos+i]].mol_feats
            clq = AtomClique(idx=idx, atom_idx=atom_clusters[idx], feats=mol_feats,
                             energy=dataset_calc[nn_idx[idx]].energy, substruct=substructs[idx])
            clqs.append(clq)

        dataset[n].set_junc_mg(clqs, edges_data[n])
        pos += n_clqs_data[n]

    if path_save_file is not None:
        torch.save(dataset, path_save_file)

    return dataset
