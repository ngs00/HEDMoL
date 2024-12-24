import torch
import numpy
import pandas
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import scale
from tqdm import tqdm
from rdkit import Chem
from itertools import chain
from util.chem import get_mol_graph


class MolData:
    def __init__(self, idx, mg, mol, mol_feats=None, energy=None, y=None):
        self.idx = idx
        self.mg = mg
        self.smg = None
        self.mol = mol
        self.mol_feats = mol_feats
        self.energy = energy
        self.y = y
        self.substructs = None
        self.idx_subatoms = None

    def set_junc_mg(self, clqs, edges):
        self.smg = Data(x=torch.vstack([c.feats for c in clqs]), edge_index=edges,
                        energy=torch.vstack([c.energy for c in clqs]))

        self.substructs = list()
        self.idx_subatoms = list()
        for c in clqs:
            if c.atom_idx.shape[0] >= 3:
                self.substructs.append(c.substruct)
                self.idx_subatoms.append(c.atom_idx)

        if len(self.substructs) == 0:
            self.substructs = None
            self.idx_subatoms = None
        else:
            self.idx_subatoms = torch.cat(self.idx_subatoms, dim=0)


class DecomposedBatch:
    def __init__(self, mg, smg, substrcuts, idx_subatoms, y=None):
        self.mg = mg
        self.smg = smg
        self.substructs = substrcuts
        self.idx_subatoms = idx_subatoms
        self.y = y

    def cuda(self):
        self.mg = self.mg.cuda()
        self.smg = self.smg.cuda()
        self.y = None if self.y is None else self.y.cuda()

        if self.substructs is not None:
            self.substructs = self.substructs.cuda()


def load_calc_dataset(path_dataset, elem_attrs, idx_struct, idx_feat, idx_energy):
    data = numpy.array(pandas.read_excel(path_dataset))
    mol_feats = data[:, numpy.atleast_1d(idx_feat)]
    norm_mol_feats = scale(mol_feats)
    dataset = list()

    for i in tqdm(range(0, data.shape[0])):
        smiles = data[i, idx_struct]
        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            mol = Chem.AddHs(mol)
            mg = get_mol_graph(mol, elem_attrs)

            if mg is not None:
                dataset.append(MolData(i, mg, mol, mol_feats=norm_mol_feats[i], energy=data[i, idx_energy]))

    return dataset


def load_dataset(path_dataset, elem_attrs, idx_struct, idx_target=None):
    data = numpy.array(pandas.read_excel(path_dataset))
    dataset = list()

    for i in tqdm(range(0, data.shape[0])):
        smiles = data[i, idx_struct]
        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            mol = Chem.AddHs(mol)
            mg = get_mol_graph(mol, elem_attrs)

            if mg is not None:
                if idx_target is None:
                    target = None
                else:
                    target = torch.tensor(data[i, idx_target], dtype=torch.float).view(1, 1)

                dataset.append(MolData(i, mg, mol, y=target))

    return dataset


def get_k_folds(dataset, k, random_seed):
    if random_seed is not None:
        numpy.random.seed(random_seed)

    idx_rand = numpy.array_split(numpy.random.permutation(len(dataset)), k)
    sub_datasets = list()
    for i in range(0, k):
        sub_datasets.append([dataset[idx] for idx in idx_rand[i]])

    k_folds = list()

    for i in range(0, k):
        dataset_train = list(chain.from_iterable(sub_datasets[:i] + sub_datasets[i + 1:]))
        dataset_test = sub_datasets[i]
        k_folds.append([dataset_train, dataset_test])

    return k_folds


def collate(batch):
    mg = list()
    smg = list()
    substructs = list()
    idx_subatoms = list()
    pos_subatoms = 0
    y = list()

    for d in batch:
        mg.append(d.mg)
        smg.append(d.smg)

        if d.substructs is not None:
            substructs += d.substructs
            idx_subatoms.append(pos_subatoms + d.idx_subatoms)

        pos_subatoms += d.mg.x.shape[0]
        y.append(d.y)

    if len(substructs) == 0:
        substructs = None
        idx_subatoms = None
    else:
        substructs = Batch.from_data_list(substructs)
        idx_subatoms = torch.cat(idx_subatoms, dim=0)

    mg = Batch.from_data_list(mg)
    smg = Batch.from_data_list(smg)
    y = torch.vstack(y)

    return DecomposedBatch(mg, smg, substructs, idx_subatoms, y)
