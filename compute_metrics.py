import json

import numpy as np
from sklearn.metrics import roc_auc_score


ITERATIONS = 5


def get_true_from_file(path):
    protein_truth = {}
    with open(path) as file:
        lines = file.readlines()

    for line in lines[1:]:
        parsed_line = line.strip().split()
        protein_truth[f'{parsed_line[0]}_{parsed_line[1]}'] = int(parsed_line[2])
    return protein_truth


def compute(probas, path=None):
    protein_predictions = {}

    protein_truth = {}
    if path is not None:
        protein_truth = get_true_from_file(path)

    for pair in sorted(probas):
        protein = pair.split('_')[0]
        pair_predictions = probas[pair]

        if path is None:
            protein_truth[pair] = pair_predictions[0]
            pair_predictions = pair_predictions[1:]

        if protein in protein_predictions:
            protein_predictions[protein][pair] = pair_predictions
        else:
            protein_predictions[protein] = {pair: pair_predictions}

    protein_metrics = []
    for protein in protein_predictions:
        iteration_metrics = []
        for iteration in range(ITERATIONS):
            predicted = []
            true = []
            for pair, predictions in protein_predictions[protein].items():
                predicted.append(predictions[iteration])
                true.append(protein_truth[pair])

            iteration_metrics.append(roc_auc_score(true, predicted))
        protein_metrics.append(np.mean(iteration_metrics))
        print(iteration_metrics)
    print(np.mean(protein_metrics))


def compute_united(probas):
    protein_predictions = [[] for _ in range(ITERATIONS + 1)]
    for pair in sorted(probas):
        for iteration in range(ITERATIONS + 1):
            protein_predictions[iteration].append(probas[pair][iteration])

    metrics = []
    for iteration in protein_predictions[1:]:
        metrics.append(roc_auc_score(protein_predictions[0], iteration))
    print(metrics)
    print(f'{np.mean(metrics) :.4f}')


def main():
    path = # path to the input-file (the probas in the json-file)
    path_to_class = # path to the fold table
    with open(path) as file:
        probas = json.load(file)

    
    compute(probas, path=path_to_class) # target-centric score
    compute_united(probas) # simple score


if __name__ == '__main__':
    main()
