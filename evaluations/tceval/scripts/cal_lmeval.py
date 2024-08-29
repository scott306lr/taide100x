import json
import os
import glob

import pandas as pd
import numpy as np

from _meta_data import mmlu_subject2category, tmmluplus_subject2category


def get_metrics_tmmlu(result_path):
    data = json.load(open(result_path))['results']
    stat = {'STEM': [], "humanities": [], "social sciences": [], "other (business, health, misc.)": []}
    for k in data.keys():
        if not k.startswith('tmmluplus_fewshot-'):
            continue
        subject = k.replace('tmmluplus_fewshot-', '')
        cat = tmmluplus_subject2category[subject]
        stat[cat].append(data[k]['acc,none'])

    agg_stat = {k: np.mean(stat[k]) for k in stat}
    
    return {
        'tmmlu_acc': np.mean(list(agg_stat.values())),
        'ttqa_acc': data[[k for k in data.keys() if 'ttqav2' in k][0]]['acc,none'],
        'tmmlu_stem_acc': agg_stat['STEM'],
        'tmmlu_humanities_acc': agg_stat['humanities'],
        'tmmlu_social-sciences_acc': agg_stat['social sciences'],
        'tmmlu_other_acc': agg_stat['other (business, health, misc.)']
    }


def get_metrics_mmlu(result_path):
    data = json.load(open(result_path))['results']
    stat = {'STEM': [], "humanities": [], "social sciences": [], "other (business, health, misc.)": []}
    for k in data.keys():
        if not k.startswith('mmlu_'):
            continue
        if k in ['mmlu_stem', 'mmlu_humanities', 'mmlu_social_sciences', 'mmlu_other']:
            continue
        subject = k.replace('mmlu_', '')
        cat = mmlu_subject2category[subject]
        stat[cat].append(data[k]['acc,none'])

    agg_stat = {k: np.mean(stat[k]) for k in stat}
    return {
        'mmlu_acc': np.mean(list(agg_stat.values())),
        'mmlu_stem_acc': agg_stat['STEM'],
        'mmlu_humanities_acc': agg_stat['humanities'],
        'mmlu_social-sciences_acc': agg_stat['social sciences'],
        'mmlu_other_acc': agg_stat['other (business, health, misc.)']
    }


def get_metrics_drcd(raw_path):
    data = json.load(open(raw_path))
    em = np.mean([x['exact_match'] for x in data])
    return {
        'drcd_em': em
    }


def get_metrics_table(result_path):
    data = json.load(open(result_path))
    acc = data['results']['penguin_table']['acc,none']
    return {
        'table_acc': acc
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("folder")
    args = parser.parse_args()

    task = args.task
    folder = args.folder
    
    if task == 'penguin_table':
        metrics = get_metrics_table(os.path.join(folder, 'results.json'))
    elif task == 'tmmlu':
        metrics = get_metrics_tmmlu(os.path.join(folder, 'results.json'))
    elif task == 'mmlu':
        metrics = get_metrics_mmlu(os.path.join(folder, 'results.json'))

    result = {
        'task': task,
        'metrics': metrics
    }
    print(result)


if __name__ == '__main__':
    main()
