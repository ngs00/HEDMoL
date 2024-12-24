from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from pandas import DataFrame
from util.chem import load_elem_attrs
from util.data import *
from hedmol.knowledge_extension import assign_calc_attrs
from hedmol.model import EGC, GIN, HEDMOL
from hedmol.schnet import SchNet


# Experiment settings.
dataset_name = 'esol'
decomposition = True
n_folds = 5
batch_size = 64
dim_z_e = 16
dim_z_a = 16
init_lr = 5e-4
l2_coeff = 5e-6
n_epochs = 500
alpha = 0.0
coeff_pcr = 1.0
list_idx_test = list()
list_targets_test = list()
list_preds_test = list()
r2_scores = list()


# Load external calculation and target datasets.
if decomposition:
    elem_attrs = scale(load_elem_attrs('res/matscholar-embedding.json'))
    dataset_calc = load_calc_dataset(path_dataset='res/qm9_max6.xlsx',
                                     elem_attrs=elem_attrs,
                                     idx_struct=0,
                                     idx_feat=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                     idx_energy=14)
    dataset = load_dataset(path_dataset='../../data/chem_data/' + dataset_name + '.xlsx',
                           elem_attrs=elem_attrs,
                           idx_struct=0,
                           idx_target=1)
    dataset = assign_calc_attrs(dataset=dataset,
                                dataset_calc=dataset_calc,
                                path_save_file='save/dataset/{}.pt'.format(dataset_name))
else:
    dataset = torch.load('save/dataset/{}.pt'.format(dataset_name))


# Split the dataset into five non-duplicated subsets.
k_folds = get_k_folds(dataset, k=n_folds, random_seed=0)


# Train and evaluate HEDMoL on the leave-one-out 5-fold cross-validation.
for k in range(0, n_folds):
    dataset_train = k_folds[k][0]
    dataset_test = k_folds[k][1]
    targets_test = numpy.vstack([d.y for d in dataset_test])
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, collate_fn=collate)

    # Define a model architecture of HEDMoL.
    emb_net_a = EGC(n_node_feats=dataset[0].mg.x.shape[1],
                    n_edge_feats=dataset[0].mg.edge_attr.shape[1],
                    dim_state=dim_z_e,
                    dim_out=dim_z_a)
    # emb_net_a = SchNet(dim_out=dim_z_a, dim_state=dim_z_e)
    emb_net_e = GIN(n_node_feats=dataset[0].smg.x.shape[1], dim_out=dim_z_e)
    model = HEDMOL(emb_net_a=emb_net_a, emb_net_e=emb_net_e, dim_out=1).cuda()

    # Fit model parameters of HEDMoL on the training dataset.
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=l2_coeff)
    criterion = torch.nn.L1Loss()
    for epoch in range(0, n_epochs):
        train_loss = model.fit(loader_train, optimizer, criterion, alpha, coeff_pcr)
        # print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss))

        preds_test = model.predict(loader_test).numpy()
        test_r2 = r2_score(targets_test, preds_test)
        print('Repeat [{}/{}]\tEpoch [{}/{}]\tTrain loss: {:.4f}\tTest R2: {:.4f}'
              .format(k + 1, n_folds, epoch + 1, n_epochs, train_loss, test_r2))

    # Save the evaluation results of HEDMoL on the test dataset.
    list_idx_test.append(numpy.vstack([d.idx for d in dataset_test]))
    list_targets_test.append(targets_test)
    list_preds_test.append(model.predict(loader_test).numpy())
    r2_scores.append(r2_score(targets_test, list_preds_test[-1]))

    # Save the model parameters of HEDMoL.
    torch.save(model.state_dict(), 'save/model_{}_{}.pt'.format(dataset_name, k))


# Save the prediction results.
eval_idx_test = numpy.vstack(list_idx_test)
eval_targets_test = numpy.vstack(list_targets_test)
eval_preds_test = numpy.vstack(list_preds_test)
eval_results = numpy.hstack([eval_idx_test, eval_targets_test, eval_preds_test])
DataFrame(eval_results).to_excel('save/preds_{}.xlsx'.format(dataset_name), index=False, header=False)


# Print mean and standard deviation of the R2-scores on the test datasets.
print('Test R2-score: {:.3f} \u00B1 {:.3f}'.format(numpy.mean(r2_scores), numpy.std(r2_scores)))
