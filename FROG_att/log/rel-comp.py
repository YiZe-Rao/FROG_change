import numpy as np
import re
import os

def extract_rel_score(per_log: str) -> dict:
    rel_scores = {}
    matches = re.findall(r"'([^']+)'\s*:\s*tensor\(\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)", per_log)
    for rel, score in matches:
        rel_scores[rel] = float(score)
    return rel_scores

def get_dict():
    table_replace = {}
    dict_table = {'reported_event_totals', 'outcome_analyses', 'drop_withdrawals', 'interventions_studies', 'facilities_studies', 'sponsors_studies', 'conditions_studies',
                  'constructor_results', 'constructor_standings',
                  'user_friends', 'event_attendees', 'event_interest',
                  }
    for table in dict_table:
        table_replace[table] = table.replace('_', '-')
    return table_replace

def judge(rel: set) -> dict:
    def _replace_table(tmp: str) -> str:
        for name in table_replace:
            if name in tmp:
                tmp = tmp.replace(name, table_replace[name])
        return tmp
    stat = {}
    flag = set()
    new_rel = {}
    for r, value in rel.items():
        underscore_count = r.count('_')
        if underscore_count != 2:
            new_key = _replace_table(r)
        else:
            new_key = r
        new_rel[new_key] = value  # 将值放到新的 key 下

    for r in new_rel.keys():
        underscore_count = r.count('_')
        assert underscore_count == 2, f'Error relation format: {r}'
        if r in flag:
            continue
        u, v, w = r.split('_', 2)
        if f'{w}_{v}_{u}' in new_rel:
            stat[r] = {'→': new_rel[r], '←': new_rel[f'{w}_{v}_{u}']}
            flag.add(f'{w}_{v}_{u}')
        else:
            stat[r] = {'→': new_rel[r], '←': None}
        flag.add(r)
    return stat

def statistic(judge_res: dict) -> None:
    for k, v in judge_res.items():
        if v['←'] is not None:
            if v['→'] <= 0.5: co_occurence['node'] += 1
            else: co_occurence['edge'] += 1
            if v['←'] <= 0.5: co_occurence['node'] += 1
            else: co_occurence['edge'] += 1

            if (v['→'] <= 0.5 and v['←'] <= 0.5) or (v['→'] > 0.5 and v['←'] > 0.5):
                co_occurence_edge['same'] += 1
            else:
                co_occurence_edge['diff'] += 1
        else:
            if v['→'] <= 0.5:
                completion['node'] += 1
            else:
                completion['edge'] += 1

def read_dir_file(dir_path: str, dataset=None) -> list:
    files = []
    for root, _, filenames in os.walk(dir_path):
        if 'ablation' in root: continue
        for filename in filenames:
            if filename.endswith('.log'):
                if dataset is None:
                    files.append(os.path.join(root, filename))
                elif dataset in filename:
                    files.append(os.path.join(root, filename))
    return files

if __name__ == '__main__':
    table_replace = get_dict()
    kept = []
    file_list = read_dir_file('./', 'hm')
    for file_path in file_list:
        if 'ipynb_checkpoints' in file_path: continue
        print(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip().lower() == 'table as node/edge:':
                if i + 1 < len(lines):
                    kept.append(lines[i+1].rstrip('\n'))

    co_occurence = {'node': 0, 'edge': 0}
    co_occurence_edge = {'same': 0, 'diff': 0}
    completion = {'node': 0, 'edge': 0}

    print('--- Processing ---')
    for log in kept:
        rel_scores = extract_rel_score(log)
        judged = judge(rel_scores)
        statistic(judged)

    print('--- Statistic ---')
    print('Co-occurrence:')
    print(f"  Node-based co-occurrences: {co_occurence['node']}")
    print(f"  Edge-based co-occurrences: {co_occurence['edge']}")
    print(f"  Same direction: {co_occurence_edge['same']}")
    print(f"  Different direction: {co_occurence_edge['diff']}")
    print('Completion:')
    print(f"  Node-based completions: {completion['node']}")
    print(f"  Edge-based completions: {completion['edge']}")