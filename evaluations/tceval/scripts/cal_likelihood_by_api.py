import re
import json
import random

from datasets import load_dataset
import openai
from tqdm import tqdm
import tiktoken
import numpy as np

from _meta_data import tmmluplus_subject2category, mmlu_subject2category

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

HEADERS = None
ENGINE = 'gpt-35-turbo'
API_KEY = ''

openai.api_key = API_KEY


def worker(x):
    option_token_ids = enc.encode('A B C D E')
    target = 'ABCDE'[x['target']] if isinstance(x['target'], int) else x['target'].strip()
    query = x['query']
    num_option = x['num_option']

    resp = openai.ChatCompletion.create(
        engine=ENGINE,
        messages=[
            {"role": "user", "content": query},
        ],
        headers=HEADERS,
        temperature=0.01,
        logit_bias={str(token_id): 100 for token_id in option_token_ids[:num_option]},
        max_tokens=1
    )
    pred = resp['choices'][0]["message"]['content'].strip()
    return 1 if target == pred else 0


def table():
    ds = load_dataset('MediaTek-Research/TCEval-v2', 'penguin_table')['test']

    corrects = []
    for data in tqdm(ds):
        x = {
            'query': f'{data["question"]}\nA. {data["A"]}\nB. {data["B"]}\nC. {data["C"]}\nD. {data["D"]}\nE. {data["E"]}\nAnswer:',
            'target': data["answer"],
            'num_option': 5
        }
        corrects.append(worker(x))
    acc = np.mean(corrects)
    return {
        'table_acc': acc
    }


def tmmlu():
    acc_dict = {}
    for subject, _ in tmmluplus_subject2category.items():
        ds = load_dataset('MediaTek-Research/TCEval-v2', f'tmmluplus-{subject}')['test']
        corrects = []
        for data in ds:
            x = {
                'query': f'{data["question"]}\nA. {data["A"]}\nB. {data["B"]}\nC. {data["C"]}\nD. {data["D"]}\nAnswer:',
                'target': data["answer"],
                'num_option': 4
            }
            corrects.append(worker(x))
        acc = np.mean(corrects)
        acc_dict[subject] = acc
    
    stat = {'STEM': [], "humanities": [], "social sciences": [], "other (business, health, misc.)": []}
    for subject, acc in acc_dict.items():
        cat = tmmluplus_subject2category[subject]
        stat[cat].append(acc)

    agg_stat = {k: np.mean(stat[k]) for k in stat}
    return {
        'tmmlu_acc': np.mean(list(agg_stat.values())),
        'ttqa_acc': acc_dict['ttqav2'],
        'tmmlu_stem_acc': agg_stat['STEM'],
        'tmmlu_humanities_acc': agg_stat['humanities'],
        'tmmlu_social-sciences_acc': agg_stat['social sciences'],
        'tmmlu_other_acc': agg_stat['other (business, health, misc.)']
    }


def mmlu():
    acc_dict = {}
    for subject, _ in mmlu_subject2category.items():
        ds = load_dataset('hails/mmlu_no_train', subject)['test']

        corrects = []
        for data in ds:
            x = {
                'query': f'{data["question"]}\nA. {data["choices"][0]}\nB. {data["choices"][1]}\nC. {data["choices"][2]}\nD. {data["choices"][3]}\nAnswer:',
                'target': data["answer"],
                'num_option': 4
            }
            corrects.append(worker(x))
        acc = np.mean(corrects)
        acc_dict[subject] = acc
    
    stat = {'STEM': [], "humanities": [], "social sciences": [], "other (business, health, misc.)": []}
    for subject, acc in acc_dict.items():
        cat = mmlu_subject2category[subject]
        stat[cat].append(acc)

    agg_stat = {k: np.mean(stat[k]) for k in stat}
    return {
        'mmlu_acc': np.mean(list(agg_stat.values())),
        'mmlu_stem_acc': agg_stat['STEM'],
        'mmlu_humanities_acc': agg_stat['humanities'],
        'mmlu_social-sciences_acc': agg_stat['social sciences'],
        'mmlu_other_acc': agg_stat['other (business, health, misc.)']
    }

if __name__ == '__main__':
    print(table())
    print(tmmlu())
    print(mmlu())
