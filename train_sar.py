import json
import pandas as pd
import numpy as np
import joblib
import datetime
import xgboost as xgb
from utils import prepare_folds
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, matthews_corrcoef


N_CORES = joblib.cpu_count()


def evaluate_model(model, test_x, test_y, fold_number=None, iteration_number=None, interactive=False):
    labels = model.predict(test_x)
    probas = model.predict_proba(test_x)[:, 1]

    roc = roc_auc_score(test_y, probas)
    mcc = matthews_corrcoef(test_y, labels)

    if interactive:
        prepare_folds.out_stream(f'iteration {iteration_number} fold {fold_number}')
        prepare_folds.out_stream(f'AUC ROC = {roc}')
        prepare_folds.out_stream(f'MCC = {mcc}')
        prepare_folds.out_stream('-' * 20)
    return probas, roc


def train_iterations(data, compound_descriptors, n_iterations=5, n_folds=5, model_type='xgb'):
    then = datetime.datetime.now()

    by_protein = data.groupby('accession')
    predictions = {}
    protein_rocs = []

    for protein, group in by_protein:
        class_count = group['class'].value_counts()
        if 0 not in class_count or 1 not in class_count:
            continue
        if class_count[0] < 5 or class_count[1] < 5:
            continue

        x = group['connectivity'].to_numpy()
        y = group['class'].to_numpy()

        connectivities = list(x)
        x = np.array([np.array(compound_descriptors[connectivity]) for connectivity in x])

        rocs = []
        for iteration in range(n_iterations):
            split = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=iteration)
            for fold, (train, test) in enumerate(split.split(x, y)):
                train_x = np.array([x[i] for i in train])
                train_y = np.array([y[i] for i in train])
                test_x = np.array([x[i] for i in test])
                test_y = np.array([y[i] for i in test])

                fold_connectivities = [connectivities[i] for i in test]

                if model_type == 'knn':
                    model = KNeighborsClassifier(weights='distance', n_jobs=N_CORES)
                elif model_type == 'svm':
                    model = SVC(kernel='rbf', C=100, gamma='scale', probability=True)
                elif model_type == 'xgb':
                    model = xgb.XGBClassifier(n_estimators=600)
                else:
                    model = RandomForestClassifier(n_estimators=600, class_weight='balanced_subsample', n_jobs=N_CORES)
                model.fit(train_x, train_y)
                probas, roc = evaluate_model(model, test_x, test_y)
                rocs.append(roc)

                for i, proba in enumerate(probas):
                    pair = f'{protein}_{fold_connectivities[i]}'
                    if pair in predictions:
                        predictions[pair].append(float(proba))
                    else:
                        predictions[pair] = [float(proba)]
        protein_rocs.append(np.median(rocs))
        prepare_folds.out_stream(f'protein {protein}')
        prepare_folds.out_stream(f'median AUC ROC = {np.median(rocs)}')
        prepare_folds.out_stream('-' * 20)
    prepare_folds.out_stream(f'time: {datetime.datetime.now() - then}')
    prepare_folds.out_stream(f'mean model AUC ROC = {np.mean(protein_rocs)}')
    prepare_folds.out_stream('-' * 20)
    return predictions


def train_qsar(grid):
    for parameters in ParameterGrid(grid):
        path = # path to the fold table (though folds are ignored for SAR)
        data = pd.read_csv(path, sep='\t')

        compound_path = # path to the compound descriptors (json-file)
        with open(compound_path) as file:
            compound_descriptors = json.load(file)

        predictions = train_iterations(data, compound_descriptors)

        output = f"{parameters['family']}_{parameters['compounds']}_xgb.json"
        with open(output, 'w') as file:
            json.dump(predictions, file)


def main():
    grid = {'family': ['all_GPCR', 'PK'], 'compounds': ['ECFPs']}
    train_qsar(grid)


if __name__ == '__main__':
    main()
