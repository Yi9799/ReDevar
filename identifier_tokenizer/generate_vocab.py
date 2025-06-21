import argparse
import os
import json
import sys

import jsonlines as jsonl
import re

from collections import defaultdict
from tqdm import tqdm


def calculate_distribution(data, dataset_type):
    var_distrib = defaultdict(int)
    for each in tqdm(data):
        func = each['norm_func']
        pattern = "@@\w+@@\w+@@"
        if dataset_type == 'varcorpus':
            if each.get('type_stripped_norm_vars') is None:
                vars_map = each.get('vars_map')
                norm_var_type = {}
                if vars_map:
                    for pair in vars_map:
                        norm_var = pair[1]
                        var = pair[0]
                        if norm_var in norm_var_type and each["type_stripped_vars"][
                            var] != 'dwarf':
                            norm_var_type[norm_var] = 'dwarf'
                        else:
                            norm_var_type[norm_var] = each["type_stripped_vars"][var]
                each['type_stripped_norm_vars'] = norm_var_type
            dwarf_norm_type = each['type_stripped_norm_vars']
        for each_var in list(re.finditer(pattern,func)):
            s = each_var.start()
            e = each_var.end()
            var = func[s:e]
            orig_var = var.split("@@")[-2]


            if orig_var in dwarf_norm_type:
                var_distrib[orig_var]+=1


    sorted_var_distrib = sorted(var_distrib.items(), key = lambda x : x[1], reverse=True)

    with open(os.path.join(args.output_dir, 'frequnecy.json'), 'w', encoding='utf-8') as f:
        json.dump(sorted_var_distrib, f)

    return sorted_var_distrib




def build_vocab(data):
    times=0

    vocab_list = []

    for idx, each in tqdm(enumerate(data)):
        if len(vocab_list) == args.vocab_size:
            print("limit reached:", args.vocab_size, "Missed:",len(data)-idx-1)
            break
        if each[0] in vocab_list:
            continue
        else:
            vocab_list.append(each[0])#


            times+=1

    idx2word, word2idx = {}, {}
    for i,each in enumerate(vocab_list):
        idx2word[i] = each
        word2idx[each] = i
    with open(os.path.join(args.output_dir, 'vocab_list.txt'), 'w', encoding='utf-8') as w:
        for i, each in enumerate(vocab_list):
            w.write(each)
            w.write("\n")
    return idx2word, word2idx

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', type=str )
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--vocab_size', type=int )
    args = parser.parse_args()



    train_data = []
    with jsonl.open(args.train_file) as ofd:
        for each in tqdm(ofd):
            train_data.append(each)

    # TODO add check to
    var_distrib_train = calculate_distribution(train_data, args.dataset_type)
    print("Train data distribution", len(var_distrib_train))



    idx2word, word2idx = build_vocab(var_distrib_train)
    print("Vocabulary size", len(idx2word))

    with open(os.path.join(args.output_dir, 'idx_to_word.json'), 'w',encoding='utf-8') as w:
        w.write(json.dumps(idx2word))
    with open(os.path.join(args.output_dir, 'word_to_idx.json'), 'w',encoding='utf-8') as w:
        w.write(json.dumps(word2idx))

