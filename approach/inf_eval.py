import os
import sys
import json
import pprint
import logging
import argparse
import jsonlines
from collections import defaultdict
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from transformers import (
    PreTrainedTokenizer,
    RobertaTokenizerFast,
)


from model import RobertaForMaskedLMv2
from transformers import RobertaConfig


l = logging.getLogger('model_main')

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "roberta": (RobertaConfig, RobertaForMaskedLMv2, RobertaTokenizerFast),
}

fids_list = []



def var_match(out, idx_to_word):
    joint_result = defaultdict(list)
    joint_result_word = defaultdict(list)
    total_dwarf = 0
    total_matched = 0
    total_matched_nooov = 0
    total_nooov = 0
    total_oov = 0


    for each in tqdm(out):
        total_vars = each["num_vars"]
        fids = each["fids"]
        pos = each["varpos"]

        pred = each["pred"]
        gold = each["gold"]


        for each_var in pos:
            if each_var == "-100":
                continue
            single_gold = set([gold[each_pos] for each_pos in
                               pos[each_var]])
            assert len(single_gold) == 1
            single_gold = list(single_gold)[0]

            all_pred_each_pos = {}
            for each_pos in pos[each_var]:
                if pred[each_pos] not in all_pred_each_pos:
                    all_pred_each_pos[pred[each_pos]] = {'count': 1  }
                else:
                    all_pred_each_pos[pred[each_pos]]['count'] += 1

            multi_pred = []
            for each in all_pred_each_pos:
                multi_pred.append((each, all_pred_each_pos[each]['count']))

            sorted_multi_pred = sorted(multi_pred, key=lambda x: x[1], reverse=True)
            single_pred = sorted_multi_pred[0][0]
            gold_word = idx_to_word.get(str(single_gold), "UNK")
            pred_word = idx_to_word.get(str(single_pred), "UNK")
            joint_result_word[fids].append((each_var, gold_word, pred_word))
            total_dwarf += 1
            if gold_word == "UNK":
                total_oov += 1
            else:
                total_nooov += 1
            if gold_word == pred_word:
                total_matched += 1
            if gold_word == pred_word and gold_word != "UNK":
                total_matched_nooov += 1

    varlevel_results = {'TOTAL_VARS': total_dwarf,
                        'TOTAL_OOV': total_oov,
                        'TOTAL_NO_OOV': total_nooov,
                        'MATCHED': total_matched,
                        'MATCHED_NO_OOV': total_matched_nooov,
                        'MATCHED_TOTAL_PCT': round(total_matched / total_dwarf, 6),
                        'MATCHED_NO_OOV_TOTAL_PCT': round(total_matched_nooov / total_dwarf, 6),
                        'MATCHED_EXCLUDE_OOV_PCT': round(total_matched_nooov / total_nooov, 6),

                        }

    json.dump(joint_result_word, open(os.path.join(args.model_name, args.prefix + "_name_predictions.json"), 'w'))
    print("varlevel_results",varlevel_results)
    return varlevel_results


class CMLDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str,block_size):
        assert os.path.isfile(file_path)
        logger.info("Creating features from dataset file at %s", file_path)

        self.examples = []
        with jsonlines.open(file_path, 'r') as f:
            for ix, line in tqdm(enumerate(f), desc="Reading Jsonlines", ascii=True):
                if (None in line["inputids"]) or (
                        None in line["labels"]):
                    print("LineNum:", ix)
                    continue
                else:
                    self.examples.append(line)

        self.block_size = int(block_size)
        self.tokenizer = tokenizer
        self.truncated = {}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        block_size = self.block_size
        tokenizer = self.tokenizer

        item = self.examples[i]

        input_ids = item["inputids"]
        labels = item["labels"]
        subword_labels = item['subword_labels']
        dataorigin_labels = item['dataorigin_labels']


        fids = item['fid']
        varmap_position = item['varmap_position']

        gold_texts = item['gold_texts']

        fids_list.append([fids,
                         len([each for each in input_ids if each == tokenizer.mask_token_id]),
                         len(input_ids),
                         varmap_position,
                         gold_texts,
                         ])


        assert len(input_ids) == len(labels)
        assert len(input_ids) == len(subword_labels)
        assert len(input_ids) == len(dataorigin_labels)

        if len(input_ids) > block_size - 2:
            self.truncated[i] = 1
        if len(input_ids) >= block_size - 2:
            input_ids = input_ids[0:block_size - 2]
            labels = labels[0:block_size - 2]
            subword_labels = subword_labels[0:block_size - 2]
            dataorigin_labels = dataorigin_labels[0:block_size - 2]
        elif len(input_ids) < block_size - 2:
            input_ids = input_ids + [tokenizer.pad_token_id] * (self.block_size - 2 - len(input_ids))
            labels = labels + [tokenizer.pad_token_id] * (self.block_size - 2 - len(labels))
            subword_labels = subword_labels + [tokenizer.pad_token_id] * (self.block_size - 2 - len(subword_labels))
            dataorigin_labels = dataorigin_labels + [tokenizer.pad_token_id] * (self.block_size - 2 - len(dataorigin_labels))

        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        labels = tokenizer.build_inputs_with_special_tokens(labels)
        subword_labels = [-100] + subword_labels + [-100]
        dataorigin_labels = [-100] + dataorigin_labels + [-100]

        assert len(input_ids) == len(labels)
        assert len(input_ids) == len(subword_labels)
        assert len(input_ids) == block_size
        assert len(input_ids) == len(dataorigin_labels)
        try:
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)
            subword_labels = torch.tensor(subword_labels, dtype=torch.long)
            dataorigin_labels = torch.tensor(dataorigin_labels, dtype=torch.long)
            mask_idxs = (input_ids == tokenizer.mask_token_id).bool()
            labels[~mask_idxs] = -100
            labels = labels.reshape(labels.size()[0], 1)
            subword_labels = subword_labels.reshape(subword_labels.size()[0], 1)
            dataorigin_labels = dataorigin_labels.reshape(dataorigin_labels.size()[0], 1)
            labels = torch.cat((labels, subword_labels,dataorigin_labels), -1)
        except:
            l.error(f"Unexpected error at index {i}: {sys.exc_info()[0]}")
            raise

        return input_ids, labels, fids


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    default="model",
    type=str,
    help="The model checkpoint for weights initialization.",
)
parser.add_argument(
    "--tokenizer_name",
    default="tokenizer",
    type=str,
    help="The tokenizer",
)
parser.add_argument(
    "--data_file",
    default="cmlm_input_ida_ft_test_latest.json",
    type=str,
    help="Input Data File to Score",
)

parser.add_argument(
    "--prefix",
    default="test",
    type=str,
    help="prefix to separate the output files",
)
parser.add_argument(
    "--batch_size",
    default=4,
    type=int,
    help="Eval Batch Size",
)

parser.add_argument(
    "--pred_path",
    default="outputs",
    type=str,
    help="path where the predictions will be stored",
)

parser.add_argument(
    "--out_vocab_map",
    default="out_vocab_file",
    type=str,
    help="path where the mapping of idx_to_word is present",
)

parser.add_argument(
    "--block_size",
    default=800,
    type=int,
    help="Optional input sequence length after tokenization."
         "The training dataset will be truncated in block of this size for training."
         "Default to the model max input length for single sentence inputs (take into account special tokens).",
)
args = parser.parse_args()

device = torch.device("cuda")
n_gpu = torch.cuda.device_count()
config_class, model_class, tokenizer_class = MODEL_CLASSES["roberta"]
config = config_class.from_pretrained(args.model_name)

if args.tokenizer_name:
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
elif args.model_name:
    tokenizer = tokenizer_class.from_pretrained(args.model_name)

idx_to_word = json.load(open(args.out_vocab_map))
config.out_vocab_size = len(idx_to_word) + 1
args.out_vocab_size = len(idx_to_word) + 1
(config.dir, filename) = os.path.split(args.train_data_file)
config.alpha=args.alpha
config.beta=args.beta

model = model_class.from_pretrained(
    args.model_name,
    from_tf=bool(".ckpt" in args.model_name),
    config=config,
)

model.to(device)
tiny_dataset = CMLDataset(tokenizer, file_path=args.data_file, block_size=args.block_size)
eval_sampler = SequentialSampler(tiny_dataset)
eval_dataloader = DataLoader(tiny_dataset, sampler=eval_sampler, batch_size=args.batch_size, shuffle=False)

model.eval()

eval_loss = 0.0
nb_eval_steps = 0

matched = {1: 0, 3: 0, 5: 0, 10: 0}
matched_not_va = {1: 0, 3: 0, 5: 0, 10: 0}
matched_dwarf = {1: 0, 3: 0, 5: 0, 10: 0}
matched_dwarf_nonsingle = {1: 0, 3: 0, 5: 0, 10: 0}
totaldecomp = {1: 0, 3: 0, 5: 0, 10: 0}
totaldwarf = {1: 0, 3: 0, 5: 0, 10: 0}
totaldwarf_nonsingle = {1: 0, 3: 0, 5: 0, 10: 0}
matched_va = {1: 0, 3: 0, 5: 0, 10: 0}
matched_oov = {1: 0, 3: 0, 5: 0, 10: 0}
totalmasked = {1: 0, 3: 0, 5: 0, 10: 0}
total_oov = {1: 0, 3: 0, 5: 0, 10: 0}

pred_list = {
    1: [],
    3: [],
    5: [],

}
gold_list = []
gold_list_subword = []
pred_list_subword = []
gold_list_dataorigin = []
pred_list_dataorigin = []
result_metrics = {"VARNAME": {"TOP1": 0,
                              "TOP3": 0,
                              "TOP5": 0,

                              "TOTAL_MASKED": {'1': 0, '3': 0, '5': 0},
                              "TOTAL_MATCHED": {'1': 0, '3': 0, '5': 0},
                              "TOTAL_DWARF": {'1': 0, '3': 0, '5': 0},
                              "TOTAL_OOV": {'1': 0, '3': 0, '5': 0},
                              "DWARF_MATCHED_PCT": {'1': 0, '3': 0, '5': 0},
                              "DWARF_MATCHED_OOV_PCT": {'1': 0, '3': 0, '5': 0},
                              "DWARF_NO_MATCHED_OOV_PCT": {'1': 0, '3': 0, '5': 0},
                              "TOTAL_MATCHED_OOV_PCT": {'1': 0, '3': 0, '5': 0},
                              },
                  "VARLEVEL_NAME": {},

                  "MISC": {"Truncated": 0,
                           "perplexity": 0,
                           "loss": 0
                           }
                  }

for batch in tqdm(eval_dataloader, desc="Evaluating"):
    inputs, labels, fids = batch

    only_masked = inputs == tokenizer.mask_token_id
    masked_gold = labels[only_masked]
    inputs = inputs.to(device)
    labels = labels.to(device)
    masked_gold_name = masked_gold[:, 0]
    gold_list.append(masked_gold_name.tolist())


    with torch.no_grad():
        lm_loss, inference, _ ,_ = model(inputs, labels=labels,fids=fids,flag="test")
        eval_loss += lm_loss.mean().item()
        masked_predict = inference.indices.cpu()[only_masked]

        for k in [1, 3, 5]:
            topked = masked_predict[:, 0:k]
            pred_list[k].append(topked.tolist())
            for i, goldtok in enumerate(masked_gold_name):
                totalmasked[k] += 1
                if goldtok.item() == -100:
                    totaldecomp[k] += 1
                    continue
                totaldwarf[k] += 1
                if goldtok.item() == args.out_vocab_size - 1: total_oov[k] += 1

                each_prediction = topked[i]
                if goldtok in each_prediction:
                    matched[k] += 1
                    matched_dwarf[k] += 1
                    if goldtok.item() == args.out_vocab_size - 1: matched_oov[k] += 1

    nb_eval_steps += 1


eval_loss = eval_loss / nb_eval_steps
perplexity = torch.exp(torch.tensor(eval_loss))
result_metrics['MISC']["perplexity"] = perplexity.item()
result_metrics['MISC']["loss"] = round(eval_loss, 2)

for i in [1, 3, 5]:

    result_metrics['VARNAME']["TOTAL_MASKED"][str(i)] = totalmasked[i]
    result_metrics['VARNAME']['TOTAL_DWARF'][str(i)] = totaldwarf[i]
    result_metrics['VARNAME']['TOTAL_OOV'][str(i)] = total_oov[i]
    result_metrics['VARNAME']["TOTAL_MATCHED"][str(i)] = matched[i]
    result_metrics['VARNAME']["TOP" + str(i)] = round(matched[i] / totalmasked[i], 6)
    result_metrics['VARNAME']["DWARF_MATCHED_PCT"][str(i)] = round(matched_dwarf[i] / totaldwarf[i], 6)
    result_metrics['VARNAME']["DWARF_MATCHED_OOV_PCT"][str(i)] = round(matched_oov[i] / totaldwarf[i], 6)
    result_metrics['VARNAME']["DWARF_NO_MATCHED_OOV_PCT"][str(i)] = round(
        (matched_dwarf[i] - matched_oov[i]) / totaldwarf[i], 6)
    result_metrics['VARNAME']["TOTAL_MATCHED_OOV_PCT"][str(i)] = round(matched_oov[i] / totalmasked[i], 6)
result_metrics['MISC']['Truncated'] = len(tiny_dataset.truncated)


print("VARNAME STATS")

print("DWARF VAR % in TOTAL:", round(totaldwarf[1] * 100 / totalmasked[1], 2))
print("DECOMPILER VAR % in TOTAL:", round(totaldecomp[1] * 100 / totalmasked[1], 2))
print('\nclassification report')

print("Model saved at:", args.model_name)
l.debug("Model saved at:", args.model_name)


flat_g, flat_p = [], []

for i in range(len(gold_list_subword)):
    flat_g += gold_list[i]
    for e in pred_list[1][i]:
        flat_p.append(e[0])


start_idx = 0
out = []
out_name = defaultdict(list)

for each in tqdm(fids_list):
    _id = each[0]
    num_vars = each[1]
    length = each[2]
    varmap_position = each[3]
    g = [idx_to_word[str(e)] if str(e) in idx_to_word else e for e in flat_g[start_idx:start_idx + num_vars]]
    p = [idx_to_word[str(e)] if str(e) in idx_to_word else e for e in
         flat_p[start_idx:start_idx + num_vars]]

    varmap_position_all = each[4]
    out.append([num_vars,
                _id,
                varmap_position,
                flat_g[start_idx:start_idx + num_vars],
                flat_p[start_idx:start_idx + num_vars],
                varmap_position_all,
                ])
    start_idx += num_vars


final_output = {}
final_list = []
for each in tqdm(out):
    fids = each[1].split("_")[0]
    num_vars = each[0]
    varpos = each[2]
    g = each[3]
    p = each[4]
    varposall = each[5]

    if fids not in final_output:
        final_output[fids] = {'fids': fids, "num_vars": num_vars, "varpos": varpos, "gold": g, "pred": p,
                             "varposall": varposall}
    else:
        final_output[fids]["num_vars"] += num_vars
        final_output[fids]['gold'] += g
        final_output[fids]['pred'] += p



for each in final_output:
    final_list.append(final_output[each])


print("result_metrics",result_metrics)
varlevel_results = var_match(final_list, idx_to_word)

for each in varlevel_results:
    result_metrics['VARLEVEL_NAME'][each] = varlevel_results[each]

print(" \n\n ------------ Prediction Results ------------")
print(f"# Variable Name: ")
for k, v in result_metrics['VARNAME']['DWARF_NO_MATCHED_OOV_PCT'].items():
    print(f"\t\tTop {k}: {v}")


with open(os.path.join(args.model_name, args.prefix + "_results.json"), 'w') as f:
    json.dump(result_metrics, f)
