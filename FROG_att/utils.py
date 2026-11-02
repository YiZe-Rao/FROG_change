import torch
from tqdm import tqdm

def get_relation(edge_types):
    one_hop_relations = set()
    for edge_type in edge_types:
        if 'rev' not in edge_type[1]:
            one_hop_relations.add((edge_type[0], edge_type[2]))
            
    return one_hop_relations

def get_2hop_relation(relations: set):
    two_hop_relations = set()
    for src1, dst1 in relations:
        for src2, dst2 in relations:
            if src1 == src2 and dst1 != dst2:   # co-occurrence
                two_hop_relations.add(((dst1, src2, dst2), 'co-occurrence'))
                two_hop_relations.add(((dst2, src2, dst1), 'co-occurrence'))
            elif src2 == dst1 and dst2 != src1: # completion
                two_hop_relations.add(((dst2, src2, src1), 'completion'))   # dimension -> fact
    return two_hop_relations

def debug_edge(data):
    print('=' * 50)
    for edge_type in data.edge_types:
        print(edge_type, ':', data[edge_type].edge_index.shape)
        # print(data[edge_type].edge_index[:, :10])
    print('=' * 50)

def debug_edge2(data):
    print('=' * 50)
    for edge_type in data.keys():
        print(edge_type, ':', data[edge_type][0].shape)
    print('=' * 50)

def tuple_2_str(t):
    return f"{t[0]}_{t[1]}_{t[2]}"