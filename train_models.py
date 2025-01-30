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
from utils import prepare_folds

N_CORES = joblib.cpu_count()
print(N_CORES)
LOG_PATH = prepare_folds.get_log_path()


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
            out_stream(f'Обучилась модель для фолда {fold + iteration * 5 + 1}')
            print(len(train_x), len(train_y))
            out_stream(f'время обучения: {datetime.datetime.now() - fold_then}')

        roc = roc_auc_score(iteration_true, iteration_probas)
        mcc = matthews_corrcoef(iteration_true, iteration_labels)
        iteration_rocs.append(roc)
        iteration_mccs.append(mcc)
        out_stream('-' * 20)
        out_stream(f'Прошла итерация {iteration + 1}')
        out_stream(f'время итерации: {datetime.datetime.now() - iteration_then}')
        out_stream(f'ROC AUC: {roc}')
        out_stream(f'MCC: {mcc}')
    return iteration_mccs, iteration_rocs, label_dict, proba_dict


def train_pcm(grid):
    for parameters in ParameterGrid(grid):
        folds = \
            rf"C:\Users\georg\OneDrive\Документы\лаба\papyrus\data_tables\{parameters['family']}s_domains_fold_table.tsv"
        compounds = \
            rf"C:\Users\georg\OneDrive\Документы\лаба\papyrus\compounds\{parameters['family']}s_{parameters['compound']}.txt"
        proteins = rf"C:\Users\georg\OneDrive\Документы\лаба\papyrus\proteins\papyrus_{parameters['protein']}.json"

        folds = pd.read_csv(folds, sep='\t')
        with open(compounds) as file:
            compounds = json.load(file)
        with open(proteins) as file:
            proteins = json.load(file)

        out_stream('файлы открыты')

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
        out_stream(f'семейство: {parameters["family"]}')
        out_stream(f'дескриптор белка: {parameters["protein"]}')
        out_stream(f'дескриптор лиганда: {parameters["compound"]}')
        out_stream(f'медиана ROC: {np.median(iteration_rocs)}')
        out_stream(f'среднее ROC: {np.mean(iteration_rocs)}')
        out_stream(f'доверительный интервал ROC: {np.std(iteration_rocs) * 2}')
        out_stream(f'медиана MCC: {np.median(iteration_mccs)}')
        out_stream(f'среднее MCC: {np.mean(iteration_mccs)}')
        out_stream(f'доверительный интервал MCC: {np.std(iteration_mccs) * 2}')
    out_stream('-' * 20)


def get_metrics(grid, plot=False, step=30):
    def get_dataframe(protein_prediction, protein_metrics, label, family):
        if label == 'QSAR':
            with open(f'../{family}_qsar_new.json') as file:
                data = json.load(file)
            protein_metrics = {protein: metrics['roc'] for protein, metrics in data.items()}
            protein_representations = {protein: metrics['amount']
                                       for protein, metrics in data.items()}
        else:
            protein_representations = {protein: len(predictions[0]) for protein, predictions in protein_prediction.items()}
        # reps = sorted()
        section_borders = list(range(0, (max(protein_representations.values()) // step + 1) * step, step))
        sections_metrics = {left_border: [] for left_border in section_borders}

        for protein, metrics in protein_metrics.items():
            section = bisect.bisect_right(section_borders, protein_representations[protein])
            section = section_borders[section - 1]
            sections_metrics[section].append(metrics)

        metrics = []
        labels = []
        sections = []
        for section in section_borders:
            for section_metrics in sections_metrics[section]:
                metrics.append(section_metrics)
                sections.append(section)
                labels.append(label)
            for section_metrics in sections_metrics[section]:
                metrics.append(section_metrics)
                sections.append(section)
                labels.append(label)

        representations = [representation for representation in protein_representations.values()]
        rep_labels = [label] * len(representations)
        families = [family] * len(metrics)
        rep_families = [family] * len(representations)

        return (pd.DataFrame({'metrics': metrics, 'section': sections, 'labels': labels, 'family': families}),
                pd.DataFrame({'reps': representations, 'labels': rep_labels,  'family': rep_families}))

    def get_metrics_for_file(parameters, metrics):
        if parameters['protein'] == 'qsar':
            return get_dataframe(None, None, 'QSAR', parameters['family'])
        path = f"{parameters['family']}_{parameters['protein']}_{parameters['compound']}_{metrics}.json"
        with open(path) as file:
            predictions = json.load(file)

        proteins = set()
        for pair in predictions:
            proteins.add(pair.split('_')[0])

        protein_prediction = {protein: [[] for _ in range(6)] for protein in proteins}
        for pair, values in predictions.items():
            for i, value in enumerate(values):
                protein_prediction[pair.split('_')[0]][i].append(value)

        protein_metrics = {}
        for protein, predictions in protein_prediction.items():
            if metrics == 'probas':
                protein_metrics[protein] = np.median([roc_auc_score(predictions[0], y_predicted)
                                                      for y_predicted in predictions[1:]])
            else:
                protein_metrics[protein] = np.median([matthews_corrcoef(predictions[0], y_predicted)
                                                      for y_predicted in predictions[1:]])

        if plot:
            return get_dataframe(protein_prediction, protein_metrics, label='PCM', family=parameters['family'])

        metric_list = list(protein_metrics.values())
        if metrics == 'probas':
            out_stream('-' * 20)
            out_stream(f'семейство: {parameters["family"]}')
            out_stream(f'дескриптор белка: {parameters["protein"]}')
            out_stream(f'дескриптор лиганда: {parameters["compound"]}')
            out_stream(f'медиана ROC: {np.median(metric_list)}')
            out_stream(f'среднее ROC: {np.mean(metric_list)}')
            out_stream(f'доверительный интервал ROC: {np.std(metric_list)}')
        else:
            out_stream(f'медиана MCC: {np.median(metric_list)}')
            out_stream(f'среднее MCC: {np.mean(metric_list)}')
            out_stream(f'доверительный интервал MCC: {np.std(metric_list)}')

    def coordinate():
        line_dfs = []
        hist_dfs = []
        for parameters in ParameterGrid(grid):
            # for parameters in [{'family': 'PK', 'compound': 'ECFPs', 'protein': 'PK_sites_gap_tolerance_0000'}, {'family': 'PK', 'compound': 'ECFPs', 'protein': 'UniRep'}, {'family': 'PK', 'compound': 'ECFPs', 'protein': 'qsar'}, ]:
            # for metrics in ['probas', 'labels']:
            #     line_df, hist_df = get_metrics_for_file(parameters, metrics)
            #     dfs.append(line_df)
            line_df, hist_df = get_metrics_for_file(parameters, 'probas')
            line_dfs.append(line_df)
            hist_dfs.append(hist_df)
        pd.concat(line_dfs).to_csv('../plot/line_df.tsv', sep='\t', index=False)
        pd.concat(hist_dfs).to_csv('../plot/hist_df.tsv', sep='\t', index=False)

    coordinate()
    out_stream('-' * 20)


def main():
    grid = {'family': ['P', 'NR', 'all_GPCR', 'PK'], 'compound': ['ECFPs'], 'protein': ['UniRep', 'qsar']}
    # get_metrics(grid, plot=True)
    train_pcm(grid)


if __name__ == '__main__':
    main()

print('egor')
