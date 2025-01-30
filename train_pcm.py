import bisect
import datetime
import re
from domain_analysis import make_fasta
import numpy as np
from sklearn.model_selection import ParameterGrid
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import json
from matplotlib import gridspec
from matplotlib import pyplot as plt
import seaborn as sns
import joblib

N_CORES = joblib.cpu_count()
print(N_CORES)
LOG_PATH = # log path


def out_stream(s):
    with open(LOG_PATH, 'a+') as log:
        log.write(s if s[-1] == '\n' else s + '\n')
    print(s if s[-1] != '\n' else s[:-1])


def train_rf(train_x, train_y):
    model = RandomForestClassifier(n_estimators=600, class_weight='balanced_subsample', n_jobs=N_CORES)
    model.fit(train_x, train_y)
    return model


def get_everything_for_model(iteration, fold, folds, proteins, compounds):
    test_compounds = list(folds['connectivity'][folds[f'iteration {iteration}'] == fold])
    test_proteins = list(folds['accession'][folds[f'iteration {iteration}'] == fold])
    test_x = list([compounds[test_compounds[i]] + proteins[test_proteins[i]]
                   for i in range(len(test_proteins))])
    test_y = list(folds['class'][folds[f'iteration {iteration}'] == fold])
    train_compounds = list(folds['connectivity'][folds[f'iteration {iteration}'] != fold])
    train_proteins = list(folds['accession'][folds[f'iteration {iteration}'] != fold])
    train_x = list([compounds[train_compounds[i]] + proteins[train_proteins[i]]
                    for i in range(len(train_proteins))])
    train_y = list(folds['class'][folds[f'iteration {iteration}'] != fold])
    return train_x, train_y, test_x, test_y, test_compounds, test_proteins


def coordinate_folds(n_iteration, folds, compounds, proteins):
    iteration_rocs = []
    iteration_mccs = []
    label_dict = {}
    proba_dict = {}
    for iteration in range(n_iteration):
        iteration_then = datetime.datetime.now()
        iteration_labels = []
        iteration_probas = []
        iteration_true = []
        for fold in range(5):
            fold_then = datetime.datetime.now()
            train_x, train_y, test_x, test_y, test_compounds, test_proteins = get_everything_for_model(iteration,
                                                                                                       fold,
                                                                                                       folds,
                                                                                                       proteins,
                                                                                                       compounds)
            model = train_rf(train_x, train_y)
            labels = model.predict(test_x)
            probas = model.predict_proba(test_x)[:, 1]
            for pair in range(len(labels)):
                pair_name = f'{test_proteins[pair]}_{test_compounds[pair]}'
                if pair_name in label_dict:
                    label_dict[pair_name].append(int(labels[pair]))
                    proba_dict[pair_name].append(float(probas[pair]))
                else:
                    label_dict[pair_name] = [test_y[pair], int(labels[pair])]
                    proba_dict[pair_name] = [test_y[pair], float(probas[pair])]
            iteration_labels += [int(label) for label in labels]
            iteration_probas += [float(proba) for proba in probas]
            iteration_true += test_y
            out_stream('-' * 20)
            out_stream(f'fold {fold + iteration * 5 + 1}')
            print(len(train_x), len(train_y))
            out_stream(f'time: {datetime.datetime.now() - fold_then}')

        roc = roc_auc_score(iteration_true, iteration_probas)
        mcc = matthews_corrcoef(iteration_true, iteration_labels)
        iteration_rocs.append(roc)
        iteration_mccs.append(mcc)
        out_stream('-' * 20)
        out_stream(f'iteration {iteration + 1}')
        out_stream(f'time: {datetime.datetime.now() - iteration_then}')
        out_stream(f'ROC AUC: {roc}')
        out_stream(f'MCC: {mcc}')
    return iteration_mccs, iteration_rocs, label_dict, proba_dict


def train_pcm(grid):
    for parameters in ParameterGrid(grid):
        folds = # path to the fold table
        compounds = # path to the compound descriptors (json-file)
        proteins = # path to the protein descriptors (json-file)

        folds = pd.read_csv(folds, sep='\t')
        with open(compounds) as file:
            compounds = json.load(file)
        with open(proteins) as file:
            proteins = json.load(file)

        out_stream('the files have been opened')

        n_iterations = 5
        iteration_mccs, iteration_rocs, label_dict, proba_dict = coordinate_folds(n_iterations,
                                                                                  folds,
                                                                                  compounds,
                                                                                  proteins)
        with open(f'{parameters["family"]}_{parameters["protein"]}_{parameters["compound"]}_labels.json', 'w') as file:
            json.dump(label_dict, file)
        with open(f'{parameters["family"]}_{parameters["protein"]}_{parameters["compound"]}_probas.json', 'w') as file:
            json.dump(proba_dict, file)

        out_stream('-' * 20)
        out_stream(f'family: {parameters["family"]}')
        out_stream(f'protein descriptor: {parameters["protein"]}')
        out_stream(f'compound descriptor: {parameters["compound"]}')
        out_stream(f'median ROC: {np.median(iteration_rocs)}')
        out_stream(f'mean ROC: {np.mean(iteration_rocs)}')
        out_stream(f'doubled sd ROC: {np.std(iteration_rocs) * 2}')
        out_stream(f'meadian MCC: {np.median(iteration_mccs)}')
        out_stream(f'mean MCC: {np.mean(iteration_mccs)}')
        out_stream(f'doubled sd MCC: {np.std(iteration_mccs) * 2}')
        out_stream('-' * 20)


def main():
    grid = {'family': ['P', 'NR', 'all_GPCR', 'PK'], 'compound': ['ECFPs'], 'protein': ['UniRep', 'qsar']}
    train_pcm(grid)


if __name__ == '__main__':
    main()
